# Propagating Execution Timeouts through CoordinationService

Notes on [OpenXLA PR #44429](https://github.com/openxla/xla/pull/44429).

## Problem

This change started from a distributed hang where one worker could get stuck
during GPU execution while the other workers continued waiting in NCCL
collectives.

OpenXLA already had `HangWatchdog` for detecting execution timeouts and
`CoordinationService` for coordinating failures between workers. The problem
was that these two paths were not connected.

A GPU execution can be stuck while its host process is still alive and sending
heartbeats. In that case, the coordination service may still see the worker as
alive, while the other workers wait indefinitely for a collective that will
never complete.

The main idea of this change was to connect an execution timeout detected by
`HangWatchdog` to the distributed error handling path.

## How It Works

At a high level, the timeout path now looks like this:

    GPU execution gets stuck
              |
              v
        HangWatchdog
              |
              v
    execution_timeout_handler
          /           \
         v             v
    abort local     report error
    collectives          |
                         v
                CoordinationService
                         |
                         v
                   peer workers
                         |
                         v
              collectives can fail

The timeout handler is stored in `GpuExecutableRunOptions`. When the watchdog
fires, the handler first calls `AbortCollectivesOnTaskFailure()` for the local
process and then reports the error through the distributed runtime client.

For distributed runs, the error eventually reaches
`CoordinationServiceAgent::ReportError()`. The agent reports it to the
coordination service, which can then propagate the failure to the other
workers.

This lets peers stop waiting on collectives involving the failed task instead
of remaining stuck indefinitely.

## Watchdog Lifetime

While working on the timeout path, there was another problem with the lifetime
of the watchdog.

With asynchronous execution, `ExecuteThunksImpl()` can return after work has
been submitted to the GPU, before the GPU has actually finished. A
`HangWatchdog::Guard` owned only by that function would therefore be destroyed
too early.

The watchdog lifetime was moved into `ExecutionWatchdogScope`, which is owned
by the outer execution path. This keeps the watchdog alive while the GPU work
is still running.

The implementation also separates creating the scope from arming the watchdog.
The scope can be created by the outer caller, while `Arm()` is called when
thunk execution actually begins.

## Reporting the Error

A new `ReportErrorToService` RPC was added for reporting an execution failure
to the coordination service.

The path crosses a few layers:

    HangWatchdog
        |
        v
    timeout handler
        |
        v
    DistributedRuntimeClient
        |
        v
    CoordinationServiceAgent::ReportError()
        |
        v
    ReportErrorToService RPC
        |
        v
    CoordinationService

The request includes information about the failing task and its incarnation.
This is useful because a task may have restarted, and an error from an old
incarnation should not be treated as an error from the current one.

## Aborting Collectives

Reporting the error to other workers is only part of the timeout handling.
The local process also needs to unwind its own collective state.

`AbortCollectivesOnTaskFailure()` marks the failed task as being in an error
state and reuses the existing collective failure handling path.

Collective state associated with the failed task can then become stale. This
prevents the same invalid collective state from being reused and allows later
operations to return an error instead of entering another wait.

## Tests

The change has tests for the watchdog itself, collective cleanup, and
multi-node error propagation.

One test lets `HangWatchdog` naturally reach its timeout and checks that the
execution timeout handler is invoked. This tests the watchdog-to-handler part
of the path.

The multi-node test checks a different part. It creates two distributed
clients and directly invokes node 0's execution timeout handler. It then checks
that node 0 enters an error state and waits for node 1 to observe the same
distributed failure.

The test is not creating a real GPU hang on node 0. Calling the timeout handler
directly simulates the point after a timeout has already been detected.

Together, these tests cover two separate steps:

    HangWatchdog -> timeout handler

and

    timeout handler -> CoordinationService -> peer node

## What I Learned

Working through this change helped me understand how GPU execution failure
handling crosses several layers of XLA.

In particular, timeout detection and distributed failure propagation are
tested separately. The watchdog tests verify that an execution timeout invokes
the configured handler, while the multi-node test invokes that handler
directly and verifies propagation through CoordinationService.

It also helped me understand the different roles of
`DistributedRuntimeClient`, `CoordinationServiceAgent`, and
`CoordinationService` in the distributed runtime.

