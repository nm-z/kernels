#!/usr/bin/env perl
package DecodeCounterCapture;
use strict;
use warnings;
use Cwd qw(abs_path);
use Digest::SHA;
use Fcntl qw(O_CREAT O_RDWR LOCK_EX LOCK_NB);
use File::Path qw(make_path);
use Getopt::Long qw(GetOptionsFromArray);
use IO::Select;
use IPC::Open3;
use JSON::PP;
use POSIX qw(isfinite strftime);
use Symbol qw(gensym);
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC sleep);

# Separate nvprof runs, never authoritative throughput measurements. This
# profiles the native gather benchmark from PR #43, not a resident model core.
our $JSON = JSON::PP->new->canonical->pretty;
our @GROUPS = (
    [traffic => [qw(sysmem_read_bytes sysmem_write_bytes dram_read_bytes dram_write_bytes)]],
    [transactions => [qw(sysmem_read_transactions sysmem_write_transactions dram_read_transactions dram_write_transactions)]],
    [execution => [qw(achieved_occupancy eligible_warps_per_cycle)]],
    [stalls => [qw(stall_memory_dependency stall_memory_throttle stall_sync)]],
);
our $LOCK = '/tmp/flash-sass-engine-validation.lock';

sub read_text {
    my ($path) = @_;
    open my $f, '<:raw', $path or die "open $path: $!\n";
    local $/; my $text = <$f>;
    close $f or die "close $path: $!\n";
    return defined($text) ? $text : '';
}
sub write_text {
    my ($path, $text) = @_;
    open my $f, '>:raw', $path or die "open $path: $!\n";
    print {$f} $text or die "write $path: $!\n";
    close $f or die "close $path: $!\n";
}
sub save {
    my ($path, $value) = @_;
    my $tmp = "$path.tmp.$$";
    write_text($tmp, $JSON->encode($value));
    rename $tmp, $path or die "rename $path: $!\n";
}
sub identity {
    my ($path) = @_;
    my $real = abs_path($path);
    die "not a regular file: $path\n" unless defined($real) && -f $real;
    open my $f, '<:raw', $real or die "open $real: $!\n";
    my @before = stat($f);
    my $hash = Digest::SHA->new(256); $hash->addfile($f);
    my @after = stat($f);
    close $f or die "close $real: $!\n";
    die "file changed while hashing $real\n"
        unless join(',', @before[0,1,7,9,10]) eq join(',', @after[0,1,7,9,10]);
    return {path => $real, bytes => $after[7], sha256 => $hash->hexdigest};
}
sub run_command {
    my ($argv, $seconds) = @_;
    my @actual = ('timeout', '--kill-after=10s', "${seconds}s", @$argv);
    my $err = gensym;
    my $pid = open3(my $input, my $output, $err, @actual);
    close $input;
    my $select = IO::Select->new($output, $err);
    my %which = (fileno($output) => 'stdout', fileno($err) => 'stderr');
    my %text = (stdout => '', stderr => '');
    while ($select->count) {
        for my $f ($select->can_read) {
            my $n = sysread($f, my $buf, 65536);
            if (!defined($n)) { next if $!{EINTR}; die "subprocess read: $!\n"; }
            if (!$n) { $select->remove($f); close $f; }
            else { $text{$which{fileno($f)}} .= $buf; }
        }
    }
    waitpid($pid, 0) == $pid or die "waitpid: $!\n";
    my $status = $?;
    return {argv => \@actual, exit => ($status & 127) ? 128 + ($status & 127) : $status >> 8, %text};
}
sub successful {
    my ($argv, $seconds, $path) = @_;
    my $result = run_command($argv, $seconds);
    save($path, $result);
    die "command exited $result->{exit}; see $path\n" if $result->{exit};
    return $result;
}
sub csv_fields {
    my ($line) = @_;
    $line =~ s/\r?\n\z//;
    my @fields;
    pos($line) = 0;
    while (1) {
        my $value;
        if ($line =~ /\G"((?:[^"]|"")*)"/gc) { $value = $1; $value =~ s/""/"/g; }
        elsif ($line =~ /\G([^",]*)/gc) { $value = $1; }
        else { die "malformed CSV field\n"; }
        push @fields, $value;
        last if pos($line) == length($line);
        die "malformed CSV delimiter\n" unless $line =~ /\G,/gc;
        if (pos($line) == length($line)) { push @fields, ''; last; }
    }
    return \@fields;
}
sub metric_number {
    my ($text, $metric) = @_;
    die "missing metric value\n" unless defined($text) && !ref($text);
    $text =~ s/^\s+|\s+$//g;
    die "unsupported/nonfinite metric value: $text\n"
        unless $text =~ /\A([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)(%?)\z/;
    my ($value, $percent) = (0 + $1, $2);
    die "negative/nonfinite metric value\n" unless isfinite($value) && $value >= 0;
    my $is_stall = $metric =~ /^stall_/;
    die "wrong metric unit for $metric\n" if ($is_stall && $percent ne '%') || (!$is_stall && length($percent));
    die "metric percentage out of range\n" if $is_stall && $value > 100;
    die "occupancy out of range\n" if $metric eq 'achieved_occupancy' && $value > 1;
    my $unit = $is_stall ? 'percent' : $metric =~ /_bytes$/ ? 'bytes' :
        $metric =~ /_transactions$/ ? 'transactions' : $metric eq 'achieved_occupancy' ? 'ratio' : 'warps_per_cycle';
    return {value => $value, unit => $unit, raw => $text};
}
sub parse_metrics {
    my ($text, $expected, $invocations) = @_;
    die "profiler reported unavailable/incomplete counters\n"
        if $text =~ /ERR_NVGPUCTRPERM|No kernels were profiled|No kernels profiled|not profiled|Error:/i;
    my %wanted = map {$_ => 1} @$expected;
    my (@header, %found, $device);
    for my $line (split /\n/, $text) {
        next if $line =~ /^\s*$/ || $line =~ /^==[0-9]+==/;
        next if !@header && $line !~ /"?Metric Name"?/;
        my $values = csv_fields($line);
        if (!@header) {
            @header = @$values;
            die "wrong profiler metric header\n" unless join('|', @header) eq
                'Device|Kernel|Invocations|Metric Name|Metric Description|Min|Max|Avg';
            next;
        }
        die "wrong profiler row width\n" unless @$values == @header;
        my %row; @row{@header} = @$values;
        my $name = $row{'Metric Name'};
        die "unexpected/duplicate metric $name\n" unless $wanted{$name} && !$found{$name};
        die "wrong kernel in profiler result\n" unless $row{Kernel} =~ /\Adie(?:\(\))?\z/;
        die "wrong profiler invocation count\n" unless $row{Invocations} =~ /\A[0-9]+\z/ && $row{Invocations} == $invocations;
        die "missing profiler device\n" unless length($row{Device});
        $device //= $row{Device};
        die "mixed profiler devices\n" unless $device eq $row{Device};
        my %stats = map {lc($_) => metric_number($row{$_}, $name)} qw(Min Max Avg);
        die "inconsistent metric min/avg/max\n" unless $stats{min}{value} <= $stats{avg}{value} && $stats{avg}{value} <= $stats{max}{value};
        $found{$name} = {device => $device, kernel => $row{Kernel}, invocations => 0 + $row{Invocations}, %stats};
    }
    die "missing workload counter rows\n" unless @header && keys(%found) == keys(%wanted);
    return \%found;
}
sub validate_workload {
    my ($result, $m, $host) = @_;
    die "benchmark did not verify outputs\n" unless ref($result) eq 'HASH' &&
        JSON::PP::is_bool($result->{preflight_verified_all}) && $result->{preflight_verified_all};
    for my $field (qw(nv tail iterations seed pattern)) {
        die "workload differs: $field\n" unless defined($result->{$field}) && !ref($result->{$field}) && "$result->{$field}" eq "$m->{$field}";
    }
    die "workload source/placement differs\n" unless
        ($result->{source_revision} // '') eq $m->{expected_source_revision} &&
        defined($result->{host_kv}) && !ref($result->{host_kv}) && "$result->{host_kv}" eq "$host" &&
        defined($result->{budget}) && !ref($result->{budget}) && "$result->{budget}" eq '512';
    my $selected = $m->{nv} < 512 ? $m->{nv} : 512;
    my %dimensions = (logical_keys => 4*$selected+$m->{tail},
        physical_gathered_slots => 4*($selected+($m->{tail} ? 1 : 0)),
        visible => 4*$m->{nv}+$m->{tail}, warmup_launches => 10);
    for my $field (keys %dimensions) {
        die "wrong workload count $field\n" unless defined($result->{$field}) && !ref($result->{$field}) &&
            "$result->{$field}" eq "$dimensions{$field}";
    }
    for my $field (qw(selection_changes verified_output_signature_changes)) {
        die "invalid workload change count\n" unless defined($result->{$field}) && !ref($result->{$field}) &&
            $result->{$field} =~ /\A[0-9]+\z/ && $result->{$field} < $m->{iterations};
    }
    for my $field (qw(cache_fnv1a64 selections_fnv1a64)) {
        die "missing workload identity $field\n" unless ($result->{$field} // '') =~ /\A[0-9a-f]{16}\z/;
    }
    if ($m->{pattern} eq 'changing' && $m->{nv} > 512) {
        for my $field (qw(selection_changes verified_output_signature_changes)) {
            die "unchanged changing-layout workload\n" unless defined($result->{$field}) && $result->{$field} == $m->{iterations} - 1;
        }
    }
    return $result;
}
sub workload_rows {
    my ($text, $iterations) = @_;
    my @lines = split /\n/, $text;
    die "missing workload CSV header\n" unless @lines && shift(@lines) eq
        'iteration,gpu_launch_us,selection_fnv1a64,verified_output_fnv1a64';
    die "incomplete workload CSV\n" unless @lines == $iterations;
    my @rows;
    for my $i (0 .. $#lines) {
        my $r = csv_fields($lines[$i]);
        die "invalid workload CSV row\n" unless @$r == 4 && $r->[0] eq "$i" &&
            $r->[1] =~ /\A[0-9]+(?:\.[0-9]+)?\z/ && isfinite(0+$r->[1]) && $r->[1] > 0 &&
            $r->[2] =~ /\A[0-9a-f]{16}\z/ && $r->[3] =~ /\A[0-9a-f]{16}\z/;
        # Timings are retained in the raw file, not used as throughput evidence.
        push @rows, {iteration => $i, selection => $r->[2], verified_output => $r->[3]};
    }
    return \@rows;
}
sub matched_workloads {
    my ($a, $b) = @_;
    for my $field (qw(source_revision nv tail iterations pattern seed budget logical_keys physical_gathered_slots cache_fnv1a64 selections_fnv1a64 selection_changes verified_output_signature_changes)) {
        die "unmatched host/VRAM workload: $field\n" unless defined($a->{$field}) && defined($b->{$field}) &&
            !ref($a->{$field}) && !ref($b->{$field}) && "$a->{$field}" eq "$b->{$field}";
    }
}
sub acquire_lock {
    my ($path, $wait, $poll, $note) = @_;
    sysopen(my $lock, $path, O_CREAT | O_RDWR, 0600) or die "open lock $path: $!\n";
    my $deadline = clock_gettime(CLOCK_MONOTONIC) + $wait;
    while (!flock($lock, LOCK_EX | LOCK_NB)) {
        die "flock $path: $!\n" unless $!{EWOULDBLOCK} || $!{EAGAIN};
        $note->() if $note;
        my $left = $deadline - clock_gettime(CLOCK_MONOTONIC);
        die "BLOCKED: shared GPU lock remained held for $wait seconds\n" if $left <= 0;
        sleep($left < $poll ? $left : $poll);
    }
    return $lock;
}
sub manifest {
    my ($path) = @_;
    my $m = $JSON->decode(read_text($path));
    die "wrong workload manifest schema\n" unless ref($m) eq 'HASH' &&
        ($m->{schema} // '') eq 'flash.issue10.counter-workload.v1';
    for my $key (qw(nv tail iterations seed)) {
        die "invalid workload dimension $key\n" unless defined($m->{$key}) && !ref($m->{$key}) && $m->{$key} =~ /\A[0-9]+\z/;
    }
    die "workload dimensions out of range\n" unless 1 <= $m->{nv} && $m->{nv} <= 65536 &&
        $m->{tail} <= 3 && 4 * $m->{nv} + $m->{tail} <= 262144 &&
        2 <= $m->{iterations} && $m->{iterations} <= 10000 && $m->{seed} <= 4294967295;
    die "unsupported layout pattern\n" unless ($m->{pattern} // '') =~ /\A(?:clustered|even|random|changing|captured)\z/;
    die "GPU UUID required\n" unless ($m->{gpu_uuid} // '') =~ /\AGPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\z/;
    die "full expected source revision required\n" unless ($m->{expected_source_revision} // '') =~ /\A[0-9a-f]{40}\z/;
    for my $key (qw(binary cubin profiler)) { $m->{$key} = identity($m->{$key} // '')->{path}; }
    die "binary/profiler not executable\n" unless -x $m->{binary} && -x $m->{profiler};
    die "source_files must identify the measured benchmark sources\n" unless ref($m->{source_files}) eq 'ARRAY' && @{$m->{source_files}};
    if ($m->{pattern} eq 'captured') {
        die "captured IDs and explicit provenance file required\n" unless $m->{captured_ids} && $m->{captured_ids_provenance};
        my $ids = identity($m->{captured_ids});
        my $p = $JSON->decode(read_text($m->{captured_ids_provenance}));
        die "captured ID provenance must bind bytes and identify origin\n" unless ref($p) eq 'HASH' &&
            ($p->{sha256} // '') eq $ids->{sha256} && defined($p->{bytes}) && $p->{bytes} == $ids->{bytes} &&
            ($p->{origin} // '') =~ /\A(?:generated_fixture|checkpoint_trace)\z/;
        die "checkpoint origin requires a raw trace identity\n" if $p->{origin} eq 'checkpoint_trace' &&
            (ref($p->{raw_trace}) ne 'HASH' || !$p->{raw_trace}{path} ||
             identity($p->{raw_trace}{path})->{sha256} ne ($p->{raw_trace}{sha256} // ''));
        $m->{captured_ids} = $ids->{path};
        $m->{captured_origin} = $p;
    }
    return $m;
}
sub main {
    my @args = @_;
    my ($input, $out, $sudo, $wait, $poll, $timeout) = (undef, undef, 0, 120, 30, 180);
    GetOptionsFromArray(\@args, 'manifest=s' => \$input, 'out=s' => \$out,
        'sudo!' => \$sudo, 'wait-seconds=i' => \$wait, 'poll-seconds=i' => \$poll,
        'timeout-seconds=i' => \$timeout) or die "invalid arguments\n";
    die "--manifest and new --out required\n" unless defined($input) && defined($out) && !@args;
    die "invalid time limits\n" unless 0 <= $wait && $wait <= 2700 && 1 <= $poll && $poll <= 60 && 1 <= $timeout && $timeout <= 900;
    die "output already exists\n" if -e $out || -l $out;
    make_path($out, {mode => 0700}); $out = abs_path($out);
    my $report = {schema => 'flash.issue10.counter-capture.v1', status => 'INCOMPLETE',
        started_utc => strftime('%Y-%m-%dT%H:%M:%SZ', gmtime),
        full_issue_acceptance => JSON::PP::false, scope => 'Single-die native gather counter workload, not full-model timing',
        timing_policy => 'Discard profiled elapsed rates from authoritative throughput; gather includes mapped transfer and dequantization',
        kernel_scope => 'All preflight, ten warmup, and measured native benchmark die launches; averages are already per invocation',
        source_policy => 'Embedded source revision and current source hashes are recorded, not a build/conversion attestation',
        pairs => []};
    save("$out/status.json", $report);
    my $ok = eval {
        my $m = manifest($input); $report->{manifest} = $m;
        $report->{collector_identity} = identity(__FILE__);
        $report->{manifest_identity} = identity($input);
        my @files = ($m->{binary}, $m->{cubin}, $m->{profiler}, @{$m->{source_files}});
        push @files, $m->{captured_ids}, $m->{captured_ids_provenance} if $m->{pattern} eq 'captured';
        $report->{input_identities} = [map {identity($_)} @files];
        save("$out/status.json", $report);
        my $checks = 0;
        my $lock = acquire_lock($LOCK, $wait, $poll, sub {
            my $r = run_command(['lslocks', '-o', 'PID,COMMAND,TYPE,MODE,PATH'], 15);
            save(sprintf('%s/lock-check-%03d.json', $out, ++$checks), $r);
            print "BUSY: shared GPU lock check $checks\n";
        });
        print "ACQUIRED: shared GPU lock; read-only device preflight\n";
        successful(['hostname'], 10, "$out/hostname.json");
        successful(['ps', '-eo', 'pid,ppid,etime,comm'], 15, "$out/processes.json");
        my $before = successful(['nvidia-smi', '-q', '-x'], 30, "$out/gpu-before.json");
        die "target GPU missing from full driver snapshot\n" unless $before->{stdout} =~ /<nvidia_smi_log>/ &&
            index($before->{stdout}, '<uuid>'.$m->{gpu_uuid}.'</uuid>') >= 0;
        my $apps = successful(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader'], 30, "$out/compute-apps.json");
        die "BLOCKED: target GPU already has a compute process\n" if grep {
            my $fields = csv_fields($_); $fields->[0] =~ s/^\s+|\s+$//g; $fields->[0] eq $m->{gpu_uuid}
        } grep {length} split /\n/, $apps->{stdout};
        my @env = ('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES='.$m->{gpu_uuid}, 'DIE_CUBIN='.$m->{cubin});
        push @env, 'GATHER_IDS='.$m->{captured_ids} if $m->{pattern} eq 'captured';
        # A privileged timeout can terminate its own profiler descendants.
        # The outer unprivileged timeout is a separate transport bound.
        my @prefix = ($sudo ? ('sudo', '-n') : (), 'timeout', '--kill-after=5s', "${timeout}s", @env);
        my $query = successful([@prefix, $m->{profiler}, '--devices', '0', '--query-metrics'], $timeout+10, "$out/metric-inventory.json");
        my $reference_workload;
        for my $group (@GROUPS) {
            for my $metric (@{$group->[1]}) {
                die "metric unavailable: $metric\n" unless ($query->{stdout}.$query->{stderr}) =~ /^\s*\Q$metric\E\s*:/m;
            }
            my $pair = {group => $group->[0], arms => {}};
            for my $host (0, 1) {
                my $stem = "$out/$group->[0]-host$host";
                my @argv = (@prefix, $m->{profiler}, '--devices', '0', '--kernels', 'die', '--csv',
                    '--metrics', join(',', @{$group->[1]}), '--log-file', "$stem.nvprof.csv",
                    $m->{binary}, map {"$_"} @$m{qw(nv tail iterations pattern)}, "$host", "$stem.workload.csv", "$m->{seed}");
                my $r = successful(\@argv, $timeout+10, "$stem.command.json");
                my $workload = validate_workload($JSON->decode($r->{stdout}), $m, $host);
                matched_workloads($reference_workload, $workload) if $reference_workload;
                $reference_workload //= $workload;
                my $counters = parse_metrics(read_text("$stem.nvprof.csv"), $group->[1], 2*$m->{iterations}+10);
                $pair->{arms}{$host} = {workload => $workload, metrics => $counters,
                    verified_rows => workload_rows(read_text("$stem.workload.csv"), $m->{iterations}),
                    raw_profiler => identity("$stem.nvprof.csv"), raw_workload => identity("$stem.workload.csv")};
                save("$stem.accepted.json", $pair->{arms}{$host});
                print "MEASURED: $group->[0] host=$host; native outputs and profiler rows validated\n";
            }
            matched_workloads($pair->{arms}{0}{workload}, $pair->{arms}{1}{workload});
            die "host/VRAM row identities differ\n" unless $JSON->encode($pair->{arms}{0}{verified_rows}) eq
                $JSON->encode($pair->{arms}{1}{verified_rows});
            push @{$report->{pairs}}, $pair;
            save("$out/status.json", $report);
        }
        for my $old (@{$report->{input_identities}}) {
            my $now = identity($old->{path});
            die "input changed during capture\n" unless $old->{sha256} eq $now->{sha256} && $old->{bytes} == $now->{bytes};
        }
        successful(['nvidia-smi', '-q', '-x'], 30, "$out/gpu-after.json");
        close $lock or die "close lock: $!\n";
        $report->{status} = 'PASS';
        1;
    };
    if (!$ok) {
        $report->{error} = "$@";
        $report->{status} = $report->{error} =~ /^BLOCKED:/ ? 'BLOCKED' : 'FAIL';
    }
    $report->{finished_utc} = strftime('%Y-%m-%dT%H:%M:%SZ', gmtime);
    save("$out/status.json", $report);
    print "$report->{status}: $report->{scope}\n";
    print STDERR $report->{error} if $report->{error};
    return $ok ? 0 : $report->{status} eq 'BLOCKED' ? 73 : 1;
}
unless (caller) { $| = 1; exit main(@ARGV); }
1;
