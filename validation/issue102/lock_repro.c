#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <signal.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

enum {
	LOCK_MILLISECONDS = 18000,
	PAYLOAD_MILLISECONDS = 22000,
	STARTUP_MILLISECONDS = 60000,
	FINISH_MILLISECONDS = 90000,
};

struct job {
	char id[64];
	char status[32];
	int exit_code;
	int has_exit_code;
	int signal;
	int has_signal;
};

static void sleep_milliseconds(long milliseconds)
{
	struct timespec remaining = {
		.tv_sec = milliseconds / 1000,
		.tv_nsec = (milliseconds % 1000) * 1000000L,
	};

	while (nanosleep(&remaining, &remaining) == -1 && errno == EINTR) {
	}
}

static int read_latest_job(const char *path, struct job *job)
{
	sqlite3 *db = NULL;
	sqlite3_stmt *statement = NULL;
	int result = 0;

	if (sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK)
		goto done;
	sqlite3_busy_timeout(db, 1000);
	if (sqlite3_prepare_v2(db,
		"SELECT id,status,exit_code,termination_signal FROM jobs "
		"WHERE origin='live' ORDER BY submitted_ns DESC LIMIT 1",
		-1, &statement, NULL) != SQLITE_OK)
		goto done;
	if (sqlite3_step(statement) != SQLITE_ROW)
		goto done;

	const unsigned char *id = sqlite3_column_text(statement, 0);
	const unsigned char *status = sqlite3_column_text(statement, 1);
	if (!id || !status)
		goto done;
	snprintf(job->id, sizeof(job->id), "%s", (const char *)id);
	snprintf(job->status, sizeof(job->status), "%s", (const char *)status);
	job->has_exit_code = sqlite3_column_type(statement, 2) != SQLITE_NULL;
	job->exit_code = job->has_exit_code ? sqlite3_column_int(statement, 2) : -1;
	job->has_signal = sqlite3_column_type(statement, 3) != SQLITE_NULL;
	job->signal = job->has_signal ? sqlite3_column_int(statement, 3) : -1;
	result = 1;

done:
	sqlite3_finalize(statement);
	if (db)
		sqlite3_close(db);
	return result;
}

static int has_busy_error(const char *path, const char *job_id)
{
	sqlite3 *db = NULL;
	sqlite3_stmt *statement = NULL;
	int result = 0;
	if (sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK)
		goto done;
	if (sqlite3_prepare_v2(db,
		"SELECT count(*) FROM documents d JOIN document_search f ON f.rowid=d.id "
		"WHERE d.job_id=? AND d.kind='runner_error' AND d.source='storage/capture' "
		"AND document_search MATCH ?",
		-1, &statement, NULL) != SQLITE_OK)
		goto done;
	sqlite3_bind_text(statement, 1, job_id, -1, SQLITE_TRANSIENT);
	sqlite3_bind_text(statement, 2,
		"sqlite3 AND busyexception AND database AND locked", -1, SQLITE_STATIC);
	if (sqlite3_step(statement) == SQLITE_ROW)
		result = sqlite3_column_int(statement, 0) > 0;

done:
	sqlite3_finalize(statement);
	if (db)
		sqlite3_close(db);
	return result;
}

static void lock_writer(const char *db_path, int ready_fd)
{
	sqlite3 *db = NULL;
	char *error = NULL;
	char ready = 'E';
	int success = 0;
	int rc = sqlite3_open(db_path, &db);
	if (rc == SQLITE_OK) {
		sqlite3_busy_timeout(db, 2000);
		rc = sqlite3_exec(db, "BEGIN IMMEDIATE", NULL, NULL, &error);
		if (rc == SQLITE_OK) {
			ready = 'R';
			success = 1;
		}
	}
	(void)write(ready_fd, &ready, 1);
	close(ready_fd);
	if (ready == 'R') {
		sleep_milliseconds(LOCK_MILLISECONDS);
		success = sqlite3_exec(db, "COMMIT", NULL, NULL, NULL) == SQLITE_OK;
	}
	if (error)
		sqlite3_free(error);
	if (db)
		sqlite3_close(db);
	_exit(success ? 0 : 1);
}

static int payload(const char *marker)
{
	while (access(marker, F_OK) != 0) {
		if (errno != ENOENT) {
			perror("wait for writer lock marker");
			return 2;
		}
		sleep_milliseconds(10);
	}
	puts("issue102 payload writing while an independent SQLite writer holds the lock");
	fflush(stdout);
	sleep_milliseconds(PAYLOAD_MILLISECONDS);
	puts("issue102 payload completed after the SQLite writer released its lock");
	fflush(stdout);
	return 0;
}

static int runner_finished(pid_t pid, int *status)
{
	pid_t result = waitpid(pid, status, WNOHANG);
	return result == pid;
}

static void cleanup(const char *directory, const char *db_path,
	const char *queue_path, const char *lock_path, const char *marker_path,
	const char *job_id)
{
	char path[640];
	if (job_id && job_id[0]) {
		snprintf(path, sizeof(path), "%s/staging/%s", directory, job_id);
		(void)rmdir(path);
	}
	snprintf(path, sizeof(path), "%s/staging", directory);
	(void)rmdir(path);
	(void)unlink(lock_path);
	(void)unlink(marker_path);
	(void)rmdir(queue_path);
	snprintf(path, sizeof(path), "%s-wal", db_path);
	(void)unlink(path);
	snprintf(path, sizeof(path), "%s-shm", db_path);
	(void)unlink(path);
	(void)unlink(db_path);
	(void)rmdir(directory);
}

static int run_harness(const char *mode, const char *runner, const char *self)
{
	char directory[] = "/tmp/gpuq-issue102-XXXXXX";
	char db_path[512], queue_path[512], lock_path[512];
	char marker_path[512];
	pid_t recorder = -1, locker = -1;
	int recorder_status = 0, locker_status = 0;
	struct job job = {0};
	int have_job = 0;
	int result = 2;

	if (!mkdtemp(directory)) {
		perror("mkdtemp");
		return 2;
	}
	snprintf(db_path, sizeof(db_path), "%s/history.sqlite3", directory);
	snprintf(queue_path, sizeof(queue_path), "%s/queue", directory);
	snprintf(lock_path, sizeof(lock_path), "%s/engine.lock", directory);
	snprintf(marker_path, sizeof(marker_path), "%s/writer-ready", directory);
	if (setenv("GPUQ_QUEUE", queue_path, 1) || setenv("GPUQ_LOCK", lock_path, 1)) {
		perror("setenv");
		goto done;
	}

	recorder = fork();
	if (recorder == -1) {
		perror("fork recorder");
		goto done;
	}
	if (recorder == 0) {
		execlp("ruby", "ruby", runner, "run", "--db", db_path,
			"--exclude", "gpu,engine,trace,profiler", "--interval", "0.25",
			"--", self, "payload", marker_path, (char *)NULL);
		perror("exec ruby gpuq");
		_exit(127);
	}

	for (int elapsed = 0; elapsed < STARTUP_MILLISECONDS; elapsed += 20) {
		if (read_latest_job(db_path, &job) && strcmp(job.status, "running") == 0) {
			have_job = 1;
			break;
		}
		sleep_milliseconds(20);
	}
	if (!have_job) {
		fprintf(stderr, "GPUQ command did not reach running state\n");
		goto wait_recorder;
	}

	int ready_pipe[2];
	if (pipe(ready_pipe) == -1) {
		perror("pipe");
		goto wait_recorder;
	}
	locker = fork();
	if (locker == -1) {
		perror("fork locker");
		close(ready_pipe[0]);
		close(ready_pipe[1]);
		goto wait_recorder;
	}
	if (locker == 0) {
		close(ready_pipe[0]);
		lock_writer(db_path, ready_pipe[1]);
	}
	close(ready_pipe[1]);
	char ready = 'E';
	if (read(ready_pipe[0], &ready, 1) != 1 || ready != 'R') {
		fprintf(stderr, "independent SQLite writer could not acquire BEGIN IMMEDIATE\n");
		close(ready_pipe[0]);
		goto wait_recorder;
	}
	close(ready_pipe[0]);
	FILE *marker = fopen(marker_path, "w");
	if (!marker) {
		perror("create writer lock marker");
		goto wait_recorder;
	}
	if (fclose(marker) != 0) {
		perror("close writer lock marker");
		goto wait_recorder;
	}

wait_recorder:
	if (recorder > 0) {
		int elapsed = 0;
		while (elapsed < FINISH_MILLISECONDS) {
			if (runner_finished(recorder, &recorder_status)) {
				recorder = -1;
				break;
			}
			sleep_milliseconds(100);
			elapsed += 100;
		}
		if (recorder > 0) {
			(void)kill(recorder, SIGKILL);
			(void)waitpid(recorder, &recorder_status, 0);
			recorder = -1;
			fprintf(stderr, "GPUQ command did not finish before test deadline\n");
			goto done;
		}
	}
	if (locker > 0) {
		pid_t locker_waited;
		do {
			locker_waited = waitpid(locker, &locker_status, 0);
		} while (locker_waited == -1 && errno == EINTR);
		if (locker_waited != locker ||
			!WIFEXITED(locker_status) || WEXITSTATUS(locker_status) != 0) {
			fprintf(stderr, "independent SQLite writer did not release its transaction cleanly\n");
			locker = -1;
			goto done;
		}
		locker = -1;
	}
	if (!read_latest_job(db_path, &job)) {
		fprintf(stderr, "GPUQ job history could not be read after completion\n");
		goto done;
	}

	int recorder_exit = WIFEXITED(recorder_status) ? WEXITSTATUS(recorder_status) : -1;
	int busy_error = has_busy_error(db_path, job.id);
	int reproduced = recorder_exit == 70 && strcmp(job.status, "capture_failed") == 0 &&
		job.has_exit_code && job.exit_code == 70 && job.has_signal &&
		job.signal == SIGTERM && busy_error;
	int survived = recorder_exit == 0 && strcmp(job.status, "succeeded") == 0 &&
		job.has_exit_code && job.exit_code == 0 && !job.has_signal;
	fprintf(stdout,
		"job=%s runner_exit=%d status=%s command_exit=%d signal=%d busy_error=%d lock_ms=%d payload_ms=%d\n",
		job.id, recorder_exit, job.status, job.exit_code, job.signal, busy_error,
		LOCK_MILLISECONDS, PAYLOAD_MILLISECONDS);

	if (strcmp(mode, "reproduce") == 0) {
		if (reproduced) {
			puts("REPRODUCED: transient SQLite writer contention terminated the managed command");
			result = 0;
		} else {
			fprintf(stderr, "expected the pre-fix capture_failed outcome; it was not observed\n");
		}
	} else if (strcmp(mode, "regression") == 0) {
		if (survived) {
			puts("PASSED: the managed command survived the transient SQLite writer");
			result = 0;
		} else {
			fprintf(stderr, "FAILED: the managed command did not survive SQLite contention\n");
		}
	} else {
		fprintf(stderr, "mode must be reproduce or regression\n");
	}

done:
	if (locker > 0) {
		(void)kill(locker, SIGKILL);
		(void)waitpid(locker, &locker_status, 0);
	}
	if (recorder > 0) {
		(void)kill(recorder, SIGKILL);
		(void)waitpid(recorder, &recorder_status, 0);
	}
	cleanup(directory, db_path, queue_path, lock_path, marker_path,
		have_job ? job.id : NULL);
	return result;
}

int main(int argc, char **argv)
{
	if (argc == 3 && strcmp(argv[1], "payload") == 0)
		return payload(argv[2]);
	if (argc != 4) {
		fprintf(stderr, "usage: %s reproduce|regression GPUQ_RUNNER SELF\n", argv[0]);
		return 2;
	}
	return run_harness(argv[1], argv[2], argv[3]);
}
