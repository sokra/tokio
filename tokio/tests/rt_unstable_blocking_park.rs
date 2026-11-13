#![warn(rust_2018_idioms)]
// Wasi doesn't support threads
#![cfg(all(tokio_unstable, feature = "full", not(target_os = "wasi"), not(miri)))]

use tokio::runtime::{self, Builder, Runtime};
use tokio::task;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

/// Long enough to survive a busy CI machine, short enough to report a failure
/// instead of hanging the test run.
const TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Park,
    Unpark,
    Stop,
}

/// Records the blocking pool events of every thread.
#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<Vec<(ThreadId, Event)>>>);

impl Recorder {
    fn record(&self, event: Event) {
        self.0.lock().unwrap().push((thread::current().id(), event));
    }

    fn events(&self) -> Vec<(ThreadId, Event)> {
        self.0.lock().unwrap().clone()
    }

    fn events_of(&self, id: ThreadId) -> Vec<Event> {
        self.events()
            .into_iter()
            .filter(|(thread, _)| *thread == id)
            .map(|(_, event)| event)
            .collect()
    }

    fn threads(&self) -> Vec<ThreadId> {
        let mut threads = Vec::new();
        for (id, _) in self.events() {
            if !threads.contains(&id) {
                threads.push(id);
            }
        }
        threads
    }

    fn count(&self, event: Event) -> usize {
        self.events().iter().filter(|(_, e)| *e == event).count()
    }

    fn install(&self, builder: &mut Builder) {
        let park = self.clone();
        let unpark = self.clone();
        let stop = self.clone();
        builder
            .on_blocking_thread_park(move || park.record(Event::Park))
            .on_blocking_thread_unpark(move || unpark.record(Event::Unpark))
            .on_thread_stop(move || stop.record(Event::Stop));
    }
}

fn builders() -> Vec<Builder> {
    let mut multi = Builder::new_multi_thread();
    multi.worker_threads(2);
    vec![Builder::new_current_thread(), multi]
}

#[track_caller]
fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + TIMEOUT;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out");
        thread::sleep(Duration::from_millis(1));
    }
}

/// Waits until every thread of the blocking pool (other than the workers)
/// waits for work.
fn wait_for_idle_blocking_pool(rt: &Runtime) {
    let metrics = rt.metrics();
    wait_until(|| metrics.num_idle_blocking_threads() == metrics.num_blocking_threads());
}

/// Every unpark follows a park, before the thread runs tasks again, and idle
/// threads have parked. Threads don't unpark before they exit.
#[test]
fn park_and_unpark_are_balanced() {
    for mut builder in builders() {
        let recorder = Recorder::default();
        recorder.install(&mut builder);
        let rt = builder.build().unwrap();

        let (tx, rx) = mpsc::channel();
        for _ in 0..3 {
            // Running tasks concurrently makes the pool spawn several threads.
            let barrier = Arc::new(std::sync::Barrier::new(3));
            let handles: Vec<_> = (0..3)
                .map(|_| {
                    let barrier = barrier.clone();
                    let tx = tx.clone();
                    rt.spawn_blocking(move || {
                        barrier.wait();
                        tx.send(thread::current().id()).unwrap();
                    })
                })
                .collect();
            rt.block_on(async {
                for handle in handles {
                    handle.await.unwrap();
                }
            });
            wait_for_idle_blocking_pool(&rt);
        }
        drop(tx);
        let task_threads: Vec<ThreadId> = rx.iter().collect();

        // Idle threads that ran a task have parked, and not unparked again.
        for id in &task_threads {
            let events = recorder.events_of(*id);
            assert_eq!(events.last(), Some(&Event::Park), "{events:?}");
        }

        drop(rt);

        // After shutdown, every thread alternated park and unpark, and parked
        // last before it stopped.
        assert!(recorder.count(Event::Park) >= task_threads.len());
        for id in recorder.threads() {
            let events = recorder.events_of(id);
            let (stop, park_unpark) = events.split_last().unwrap();
            assert_eq!(*stop, Event::Stop, "{events:?}");
            assert_eq!(park_unpark.last(), Some(&Event::Park), "{events:?}");
            for (i, event) in park_unpark.iter().enumerate() {
                let expected = if i % 2 == 0 {
                    Event::Park
                } else {
                    Event::Unpark
                };
                assert_eq!(*event, expected, "{events:?}");
            }
        }
    }
}

/// The park callback runs before the thread starts waiting for work, so the
/// thread doesn't count as idle yet.
#[test]
fn park_runs_before_thread_is_idle() {
    for mut builder in builders() {
        let idle_while_parking = Arc::new(Mutex::new(Vec::new()));
        builder.on_blocking_thread_park({
            let idle_while_parking = idle_while_parking.clone();
            move || {
                let metrics = runtime::Handle::current().metrics();
                idle_while_parking
                    .lock()
                    .unwrap()
                    .push(metrics.num_idle_blocking_threads());
            }
        });
        let rt = builder.build().unwrap();

        rt.block_on(rt.spawn_blocking(|| {})).unwrap();
        wait_for_idle_blocking_pool(&rt);

        assert_eq!(*idle_while_parking.lock().unwrap(), [0]);
    }
}

/// A thread whose keep-alive expires exits without unparking.
#[test]
fn no_unpark_on_keep_alive_timeout() {
    for mut builder in builders() {
        let recorder = Recorder::default();
        recorder.install(&mut builder);
        builder.thread_keep_alive(Duration::from_millis(10));
        let rt = builder.build().unwrap();

        let id = rt
            .block_on(rt.spawn_blocking(|| thread::current().id()))
            .unwrap();
        wait_until(|| recorder.events_of(id).contains(&Event::Stop));

        assert_eq!(recorder.events_of(id), [Event::Park, Event::Stop]);
    }
}

/// The park callback runs without holding the blocking pool's lock and within
/// the runtime context, so it can spawn blocking tasks.
#[test]
fn spawn_blocking_from_park_callback() {
    for mut builder in builders() {
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(Some(tx));
        builder.on_blocking_thread_park(move || {
            if let Some(tx) = tx.lock().unwrap().take() {
                task::spawn_blocking(move || tx.send(()).unwrap());
            }
        });
        let rt = builder.build().unwrap();

        rt.block_on(rt.spawn_blocking(|| {})).unwrap();

        rx.recv_timeout(TIMEOUT).unwrap();
    }
}

/// A task spawned by the park callback, while the only blocking thread can't be
/// woken up because it doesn't wait yet, is picked up when the callback
/// returns. It must not wait for a wakeup that is never sent, which would delay
/// it until the keep-alive expires.
#[test]
fn task_queued_by_park_callback_runs_promptly() {
    for mut builder in builders() {
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(Some(tx));
        builder
            // The worker threads of the multi-thread runtime don't count.
            .max_blocking_threads(1)
            .thread_keep_alive(Duration::from_secs(60))
            .on_blocking_thread_park(move || {
                if let Some(tx) = tx.lock().unwrap().take() {
                    task::spawn_blocking(move || tx.send(thread::current().id()).unwrap());
                }
            });
        let rt = builder.build().unwrap();

        let id = rt
            .block_on(rt.spawn_blocking(|| thread::current().id()))
            .unwrap();

        // The task ran on the same thread, since there is only one.
        assert_eq!(rx.recv_timeout(TIMEOUT).unwrap(), id);
    }
}

/// A task spawned while the only blocking thread runs the park callback is
/// picked up when the callback returns. It must not wait for a wakeup that was
/// never sent, which would delay it until the keep-alive expires.
#[test]
fn task_spawned_during_park_callback_runs_promptly() {
    for mut builder in builders() {
        let (parking_tx, parking_rx) = mpsc::channel();
        let parking_tx = Mutex::new(Some(parking_tx));
        let first_park = AtomicBool::new(true);
        builder
            // The worker threads of the multi-thread runtime don't count.
            .max_blocking_threads(1)
            .thread_keep_alive(Duration::from_secs(60))
            .on_blocking_thread_park(move || {
                if first_park.swap(false, Ordering::SeqCst) {
                    parking_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                    thread::sleep(Duration::from_millis(200));
                }
            });
        let rt = builder.build().unwrap();

        rt.block_on(rt.spawn_blocking(|| {})).unwrap();
        parking_rx.recv_timeout(TIMEOUT).unwrap();

        let start = Instant::now();
        rt.block_on(rt.spawn_blocking(|| {})).unwrap();
        assert!(start.elapsed() < TIMEOUT, "took {:?}", start.elapsed());
    }
}

/// The unpark callback also works without a park callback: it is called when an
/// idle thread is woken up for a task, but not when it exits at shutdown.
#[test]
fn unpark_without_park() {
    for mut builder in builders() {
        let recorder = Recorder::default();
        let unpark = recorder.clone();
        let stop = recorder.clone();
        builder
            .max_blocking_threads(1)
            .thread_keep_alive(Duration::from_secs(60))
            .on_blocking_thread_unpark(move || unpark.record(Event::Unpark))
            .on_thread_stop(move || stop.record(Event::Stop));
        let rt = builder.build().unwrap();

        let id = rt
            .block_on(rt.spawn_blocking(|| thread::current().id()))
            .unwrap();
        wait_for_idle_blocking_pool(&rt);
        assert_eq!(recorder.events_of(id), []);

        // Wakes up the idle thread, the only one.
        let second = rt
            .block_on(rt.spawn_blocking(|| thread::current().id()))
            .unwrap();
        assert_eq!(second, id);
        assert_eq!(recorder.events_of(id), [Event::Unpark]);
        wait_for_idle_blocking_pool(&rt);

        drop(rt);
        assert_eq!(recorder.events_of(id), [Event::Unpark, Event::Stop]);
    }
}

/// Without a park callback, a thread whose keep-alive expires exits without
/// unparking.
#[test]
fn unpark_without_park_not_called_on_keep_alive_timeout() {
    for mut builder in builders() {
        let recorder = Recorder::default();
        let unpark = recorder.clone();
        let stop = recorder.clone();
        builder
            .thread_keep_alive(Duration::from_millis(10))
            .on_blocking_thread_unpark(move || unpark.record(Event::Unpark))
            .on_thread_stop(move || stop.record(Event::Stop));
        let rt = builder.build().unwrap();

        let id = rt
            .block_on(rt.spawn_blocking(|| thread::current().id()))
            .unwrap();
        wait_until(|| recorder.events_of(id).contains(&Event::Stop));

        assert_eq!(recorder.events_of(id), [Event::Stop]);
    }
}

/// The unpark callback runs without holding the blocking pool's lock and
/// within the runtime context, so it can spawn blocking tasks.
#[test]
fn spawn_blocking_from_unpark_callback() {
    for mut builder in builders() {
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(Some(tx));
        builder
            .max_blocking_threads(1)
            .thread_keep_alive(Duration::from_secs(60))
            .on_blocking_thread_unpark(move || {
                if let Some(tx) = tx.lock().unwrap().take() {
                    task::spawn_blocking(move || tx.send(()).unwrap());
                }
            });
        let rt = builder.build().unwrap();

        // The thread unparks when it is woken up for the second task.
        rt.block_on(rt.spawn_blocking(|| {})).unwrap();
        wait_for_idle_blocking_pool(&rt);
        rt.block_on(rt.spawn_blocking(|| {})).unwrap();

        rx.recv_timeout(TIMEOUT).unwrap();
    }
}

/// Registering a callback again replaces the previous one.
#[test]
fn callbacks_are_replaced() {
    for mut builder in builders() {
        let replaced = Arc::new(AtomicUsize::new(0));
        let parks = Arc::new(AtomicUsize::new(0));
        let unparks = Arc::new(AtomicUsize::new(0));
        let count = |counter: &Arc<AtomicUsize>| {
            let counter = counter.clone();
            move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        };
        builder
            .max_blocking_threads(1)
            .thread_keep_alive(Duration::from_secs(60))
            .on_blocking_thread_park(count(&replaced))
            .on_blocking_thread_park(count(&parks))
            .on_blocking_thread_unpark(count(&replaced))
            .on_blocking_thread_unpark(count(&unparks));
        let rt = builder.build().unwrap();

        // The second task wakes up the idle thread.
        rt.block_on(rt.spawn_blocking(|| {})).unwrap();
        wait_for_idle_blocking_pool(&rt);
        rt.block_on(rt.spawn_blocking(|| {})).unwrap();
        wait_for_idle_blocking_pool(&rt);
        drop(rt);

        assert_eq!(replaced.load(Ordering::SeqCst), 0);
        assert!(parks.load(Ordering::SeqCst) > 0);
        assert!(unparks.load(Ordering::SeqCst) > 0);
    }
}

/// Parking a worker of the scheduler doesn't call the blocking park callback,
/// only `on_thread_park` does.
#[test]
fn worker_park_does_not_call_blocking_park() {
    let recorder = Recorder::default();
    let worker_parks = Arc::new(AtomicUsize::new(0));
    let mut builder = Builder::new_multi_thread();
    builder.worker_threads(2).on_thread_park({
        let worker_parks = worker_parks.clone();
        move || {
            worker_parks.fetch_add(1, Ordering::SeqCst);
        }
    });
    recorder.install(&mut builder);
    let rt = builder.build().unwrap();

    for _ in 0..10 {
        rt.block_on(async {
            tokio::spawn(async { tokio::task::yield_now().await })
                .await
                .unwrap();
        });
    }
    wait_until(|| worker_parks.load(Ordering::SeqCst) > 0);

    assert_eq!(recorder.count(Event::Park), 0);
    assert_eq!(recorder.count(Event::Unpark), 0);
}
