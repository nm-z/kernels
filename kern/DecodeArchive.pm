package DecodeArchive;
use strict;
use warnings;
use JSON::PP;
use IO::Uncompress::Gunzip qw(gunzip $GunzipError);
use POSIX qw(isfinite);
use Scalar::Util qw(looks_like_number);
use File::Basename qw(dirname);

# Replay the compact, preserved issue-10 archive. This does not read the raw
# phase CSVs, establish missing launch provenance, or execute a GPU workload.
my @parts=qw(dispatch exchange_wait body_excluding_ple_wait barrier_send ple_wait);
my @components=(@parts,qw(between_phase publication_footer between_token));
my @attention=qw(attnqkq idxqk idxscore idxsel kvgather attnpartn attnmerge);
my @required=qw(traced.json.gz untraced.json.gz traced-all-die-phase-totals.json.gz
                traced-summary.json untraced-summary.json matched-phase-deltas.json);
my $json=JSON::PP->new->canonical;
sub number {
    my($x,$label,$integer)=@_;
    die "invalid number $label\n" if !defined($x)||ref($x)||!looks_like_number($x)||!isfinite(0+$x);
    die "invalid integer $label\n" if $integer && ($x<0 || int($x)!=$x);
    return $x;
}
sub close_number {
    my($actual,$expected,$label,$tolerance)=@_;
    $tolerance//=1e-8;
    number($actual,$label);number($expected,"expected $label");
    my $error=abs($actual-$expected);
    die "$label differs: $actual versus $expected\n" if $error>$tolerance;
    return $error;
}
sub compare {
    my($a,$b,$path)=@_;my $max=0;
    if(ref($a) eq 'HASH' && ref($b) eq 'HASH'){
        die "different keys at $path\n" unless join("\0",sort keys %$a) eq join("\0",sort keys %$b);
        for my $k(keys %$a){my $e=compare($a->{$k},$b->{$k},"$path/$k");$max=$e if $e>$max;}
    }elsif(ref($a) eq 'ARRAY' && ref($b) eq 'ARRAY'){
        die "different lengths at $path\n" unless @$a==@$b;
        for my $i(0..$#$a){my $e=compare($a->[$i],$b->[$i],"$path/$i");$max=$e if $e>$max;}
    }elsif(!ref($a)&&!ref($b)&&defined($a)&&defined($b)&&looks_like_number($a)&&looks_like_number($b)){
        $max=close_number($a,$b,$path);
    }else{
        die "different value at $path\n" unless $json->encode($a) eq $json->encode($b);
    }
    return $max;
}
sub read_json {
    my($path)=@_;
    return DecodeForensics::load($path) unless $path=~/\.gz\z/;
    my $compressed=DecodeForensics::read_text($path);my($text,$trailing);
    gunzip \$compressed=>\$text,Strict=>1,Transparent=>0,TrailingData=>\$trailing
      or die "gunzip $path: $GunzipError\n";
    die "trailing data after gzip $path\n" if length($trailing//'');
    return $json->decode($text);
}
sub verify_identities {
    my($root)=@_;my $ids=read_json("$root/IDENTITIES.json");
    die "identities must be an object\n" unless ref($ids) eq 'HASH';
    for my $p(@required){die "identity missing for $p\n" unless exists $ids->{$p};}
    for my $p(sort keys %$ids){
        die "unsafe identity path\n" if $p=~m{\A/|(?:\A|/)\.\.?(?:/|\z)|[\x00\r\n]};
        my $entry=$ids->{$p};my $file="$root/$p";
        die "invalid identity for $p\n" unless ref($entry) eq 'HASH' &&
          ($entry->{sha256}//'')=~/\A[0-9a-f]{64}\z/;
        number($entry->{bytes},"bytes $p",1);
        die "missing or linked artifact $p\n" unless -f $file && !-l $file;
        die "artifact size mismatch $p\n" unless (-s $file)==$entry->{bytes};
        die "artifact hash mismatch $p\n" unless DecodeForensics::sha($file) eq $entry->{sha256};
    }
    return {files=>0+keys(%$ids),identity_manifest_sha256=>DecodeForensics::sha("$root/IDENTITIES.json"),
      input_sha256=>{map {$_=>$ids->{$_}{sha256}} @required}};
}
sub key {my($r)=@_;return join(':',@$r{qw(depth placement pair)});}
sub selected_count {
    my($pos)=@_;number($pos,'selected position',1);
    my $blocks=int(($pos+1)/4);$blocks=512 if $blocks>512;
    return 4*$blocks+($pos+1)%4;
}
sub summary {
    my($runs)=@_;my %groups;
    push @{$groups{$_->{depth}}{$_->{placement}}},$_ for @$runs;
    my @out;
    for my $depth(sort {$a<=>$b} keys %groups){
        for my $place(qw(host vram)){
            my @r=sort {$a->{pair}<=>$b->{pair}} @{$groups{$depth}{$place}//[]};
            die "incomplete placement pairs\n" unless @r==5;
            my @means=map {$_->{client_decode_ms}/128} @r;
            my @intervals=map {map {$_/1e6} @{$_->{intervals_ns}}} @r;
            my $median=DecodeForensics::median(\@means);
            push @out,{depth=>0+$depth,placement=>$place,runs=>5,
              median_ms_per_token=>$median,median_tok_s=>1000/$median,
              p95_interval_ms=>DecodeForensics::p95(\@intervals),per_run_ms_per_token=>\@means};
        }
    }
    return \@out;
}
sub validate_runs {
    my($capture,$traced)=@_;
    die "wrong archive schema\n" unless ref($capture) eq 'HASH' &&
      ($capture->{schema}//'') eq 'flash-sass.decode-forensics.v1' &&
      JSON::PP::is_bool($capture->{traced}) && !!$capture->{traced}==!!$traced &&
      ($capture->{tag}//'') eq ($traced?'traced':'untraced');
    my $m=$capture->{manifest};die "missing manifest\n" unless ref($m) eq 'HASH';
    die "wrong capture dimensions\n" unless number($m->{pairs},'pairs',1)==5 &&
      number($m->{outputs},'outputs',1)==129;
    die "invalid source SHA\n" unless ($m->{source_sha}//'')=~/\A[0-9a-f]{40}\z/;
    my $cfg=$m->{configs};
    die "missing paired configurations\n" unless ref($cfg) eq 'HASH' &&
      join(',',sort keys %$cfg) eq 'host,vram';
    for my $p(qw(host vram)){
        my $c=$cfg->{$p};
        die "invalid configuration $p\n" unless ref($c) eq 'HASH' &&
          number($c->{kv_host},'kv_host',1)==($p eq 'host'?1:0) &&
          number($c->{nblk},'nblk',1)==512 && ($c->{layers}//'') eq '48/48' &&
          number($c->{dies},'dies',1)==6 &&
          number($c->{trace_tokens},'trace_tokens',1)==($traced?129:0);
        number($c->{maxt},'capacity',1);
        die "configuration source differs from manifest\n" unless ($c->{rev}//'') eq $m->{source_sha};
    }
    my %h=%{$cfg->{host}};my %v=%{$cfg->{vram}};delete $h{kv_host};delete $v{kv_host};
    die "unmatched configurations\n" unless $json->encode(\%h) eq $json->encode(\%v);
    my @depths=$traced?(256,2048,2051,2052,3100,8000):(3100,8000);
    my %depth=map {$_=>1} @depths;my %seen;my %pairs;my %runs;
    die "wrong number of runs\n" unless ref($capture->{runs}) eq 'ARRAY' &&
      @{$capture->{runs}}==@depths*10;
    for my $r(@{$capture->{runs}}){
        die "incomplete run\n" unless ref($r) eq 'HASH' && ($r->{status}//'') eq 'complete';
        my $d=number($r->{depth},'depth',1);my $p=$r->{placement}//'';
        my $pair=number($r->{pair},'pair',1);
        die "invalid run key\n" unless $depth{$d} && exists($cfg->{$p}) && $pair<5;
        my $key=key($r);die "duplicate run $key\n" if $seen{$key}++;
        die "run exceeds capacity\n" if $d+128>$cfg->{$p}{maxt};
        die "wrong output dimensions\n" unless number($r->{decode_intervals},'interval count',1)==128 &&
          ref($r->{generated_ids}) eq 'ARRAY' && @{$r->{generated_ids}}==129 &&
          ref($r->{client_observed_ns}) eq 'ARRAY' && @{$r->{client_observed_ns}}==129 &&
          ref($r->{intervals_ns}) eq 'ARRAY' && @{$r->{intervals_ns}}==128;
        for(@{$r->{generated_ids}}){die "token ID overflow\n" if number($_,'token ID',1)>4294967295;}
        my $obs=$r->{client_observed_ns};
        number($_,'observation',1) for @$obs;
        for my $i(1..128){
            my $delta=$obs->[$i]-$obs->[$i-1];
            die "nonpositive observation interval\n" unless $delta>0;
            close_number($r->{intervals_ns}[$i-1],$delta,'observation interval',0);
        }
        my $sum=DecodeForensics::total($r->{intervals_ns});
        close_number($r->{client_decode_ms},$sum/1e6,'client decode ms',1e-9);
        close_number($r->{client_ms_per_token},$sum/128e6,'client mean');
        my $start=number($r->{request_start_ns},'request start',1);
        my $end=number($r->{request_end_ns},'request end',1);
        die "observations outside request\n" unless $start<=$obs->[0] && $obs->[-1]<=$end && $end>$start;
        close_number($r->{request_ttft_ms},($obs->[0]-$start)/1e6,'request TTFT');
        close_number($r->{request_completion_ms},($end-$start)/1e6,'request completion');
        my $done=$r->{done};die "missing completion\n" unless ref($done) eq 'HASH';
        for my $field(qw(generated ended prompt resumed snapshot)){number($done->{$field},"done $field",1);}
        die "incomplete completion\n" unless $done->{generated}==129 && !$done->{ended} &&
          $done->{prompt}==$d && $done->{resumed}==$d-1;
        number($done->{run_ms},'run ms');number($done->{first_ms},'first ms');
        die "invalid completion timing\n" unless $done->{run_ms}>$done->{first_ms} && $done->{first_ms}>=0;
        close_number($r->{reported_post_first_ms},$done->{run_ms}-$done->{first_ms},'reported post-first interval');
        my @raw=split /\s+/,$r->{raw_done}//'';
        die "invalid raw completion\n" unless @raw==8 && shift(@raw) eq 'done';
        my @fields=qw(generated ended prompt run_ms first_ms resumed snapshot);
        close_number($raw[$_],$done->{$fields[$_]},"raw completion $fields[$_]",0) for 0..$#fields;
        my $order=$r->{order};
        die "invalid interleaving order\n" unless ref($order) eq 'ARRAY' && @$order==2 &&
          join(',',sort @$order) eq 'host,vram';
        my $pk="$d:$pair";
        if(my $prev=$pairs{$pk}){
            die "paired output mismatch\n" unless $json->encode($prev->{generated_ids}) eq $json->encode($r->{generated_ids});
            die "paired order mismatch\n" unless $json->encode($prev->{order}) eq $json->encode($order);
        }else{$pairs{$pk}=$r;}
        $runs{$key}=$r;
    }
    for my $d(@depths){for my $p(qw(host vram)){for my $pair(0..4){die "missing run\n" unless $seen{"$d:$p:$pair"};}}}
    my %skips;
    die "invalid capacity skips\n" unless ref($capture->{capacity_skips}) eq 'ARRAY';
    for my $s(@{$capture->{capacity_skips}}){
        my $d=number($s->{depth},'skipped depth',1);my $capacity=number($s->{capacity},'skipped capacity',1);
        die "invalid capacity skip\n" unless ($s->{status}//'') eq 'capacity_skip' &&
          !$skips{$d}++ && !$depth{$d} && $d+128>$capacity && $capacity==$cfg->{host}{maxt};
    }
    die "wrong capacity skip coverage\n" unless join(',',sort {$a<=>$b} keys %skips) eq ($traced?'65000,262000':'');
    my $summary=summary($capture->{runs});
    compare($summary,$capture->{summary},'embedded summary');
    return (\%runs,$summary);
}
sub validate_phases {
    my($phase,$runs,$configs)=@_;
    die "incomplete all-die archive\n" unless ref($phase) eq 'ARRAY' && @$phase==60;
    my %seen;my $max_residual=0;my $count=0;
    my @map=split /,/,$configs->{host}{attn_map}//'';my %owners;
    die "missing attention map\n" unless @map==12;
    for(@map){die "invalid attention owner\n" unless /\A[0-5]\z/;$owners{$_}++;}
    for my $frame(@$phase){
        my $key=key($frame);my $run=$runs->{$key};
        die "unknown or repeated phase frame\n" unless $run && !$seen{$key}++;
        die "missing dies\n" unless ref($frame->{dies}) eq 'ARRAY' && @{$frame->{dies}}==6;
        my %die;my %work;my $request;
        for my $d(@{$frame->{dies}}){
            my $id=number($d->{die},'die',1);
            die "invalid or repeated die\n" unless $id<6 && !$die{$id}++;
            my $meta=$d->{metadata};
            die "invalid trace metadata\n" unless ref($meta) eq 'HASH' &&
              number($meta->{die},'metadata die',1)==$id &&
              number($meta->{count},'trace count',1)==129 &&
              number($meta->{first},'first position',1)==$run->{depth}-1 &&
              number($meta->{nblk},'trace nblk',1)==512 &&
              number($meta->{ctas},'trace CTAs',1)==1;
            my $rid=number($meta->{request},'request ID',1);
            die "mixed request IDs\n" if defined($request)&&$request!=$rid;$request=$rid;
            my $nph=number($meta->{nph},'phase count',1);die "empty phase program\n" unless $nph;
            die "missing source summary identity\n" unless ($d->{source_summary_sha256}//'')=~/\A[0-9a-f]{64}\z/;
            my $comp=$d->{components_us_per_token};
            die "wrong timeline components\n" unless ref($comp) eq 'HASH' &&
              join(',',sort keys %$comp) eq join(',',sort @components);
            for my $c(@components){die "negative phase time\n" if number($comp->{$c},"component $c")<0;}
            my $duration=number($d->{device_us_per_token},'device duration');
            die "nonpositive device duration\n" unless $duration>0;
            my $residual=$duration-DecodeForensics::total([values %$comp]);
            close_number($residual,0,'timeline residual',1e-5);
            close_number($d->{reconciliation_residual_us},$residual,'recorded residual',1e-8);
            $max_residual=abs($residual) if abs($residual)>$max_residual;
            die "missing device publication times\n" unless ref($run->{device_publication_decode_ms_by_die}) eq 'ARRAY' &&
              @{$run->{device_publication_decode_ms_by_die}}==6;
            close_number($duration*128/1000,$run->{device_publication_decode_ms_by_die}[$id],'device publication duration');
            my $means=$d->{op_mean_us};my $counts=$d->{op_counts};
            die "invalid opcode aggregates\n" unless ref($means) eq 'HASH' && ref($counts) eq 'HASH' &&
              join(',',sort keys %$means) eq join(',',sort keys %$counts);
            my %sums;
            for my $op(keys %$means){
                my $count=number($counts->{$op},"count $op",1);
                die "invalid per-token opcode count $op\n" unless $count>0 && $count%128==0;
                die "wrong opcode components\n" unless ref($means->{$op}) eq 'HASH' &&
                  join(',',sort keys %{$means->{$op}}) eq join(',',sort @parts);
                for my $part(@parts){
                    my $v=number($means->{$op}{$part},"$op/$part");
                    die "negative opcode time\n" if $v<0;$sums{$part}+=$v;
                }
            }
            close_number(DecodeForensics::total([values %$counts]),$nph*128,'all phase counts',0);
            close_number($sums{$_}//0,$comp->{$_},"opcode sum $_") for @parts;
            for my $op(@attention){
                my $expected=($owners{$id}//0)*128*($op eq 'idxsel'?5:1);
                close_number($counts->{$op}//0,$expected,"attention count $op",0);
                $work{$op}+=DecodeForensics::total([values %{$means->{$op}//{}}]);
            }
            my $selected=$d->{selected_keys};
            die "missing selected-key counts\n" unless ref($selected) eq 'ARRAY' && @$selected==129;
            for my $i(0..128){
                die "bad selected-key row\n" unless ref($selected->[$i]) eq 'ARRAY' && @{$selected->[$i]}==2;
                my($pos,$keys)=@{$selected->[$i]};number($pos,'selected position',1);number($keys,'selected keys',1);
                die "wrong selected-key position\n" unless $pos==$run->{depth}-1+$i;
                die "wrong selected-key count\n" unless $keys==selected_count($pos);
            }
            compare($d,$run->{die0_timeline},'die0 projection') if $id==0;
            $count++;
        }
        compare(\%work,$run->{distinct_attention_layer_work_us},'distinct attention work');
        close_number($run->{client_minus_device0_us_per_token},
          $run->{client_ms_per_token}*1000-$run->{die0_timeline}{device_us_per_token},'host/device residual');
        my $engine=number($run->{engine_observed_decode_ms},'engine interval');
        die "nonpositive engine interval\n" unless $engine>0;
        close_number($run->{client_minus_engine_interval_ms},$run->{client_decode_ms}-$engine,'client/engine residual');
        close_number($run->{engine_minus_gpu0_interval_ms},$engine-$run->{device_publication_decode_ms_by_die}[0],'engine/device residual');
    }
    die "phase coverage differs from runs\n" unless keys(%seen)==keys(%$runs) && $count==360;
    return {die_frames=>$count,max_recomputed_residual_us=>$max_residual};
}
sub witness {
    my($runs)=@_;my %w;
    for my $p(qw(host vram)){
        my($a,$b)=@$runs{"3100:$p:2","8000:$p:2"};
        die "missing witness pair\n" unless $a&&$b;
        $w{$p}={pair=>2,
          client_delta_us_per_token=>1000*($b->{client_ms_per_token}-$a->{client_ms_per_token}),
          die0_delta_us_per_token=>$b->{die0_timeline}{device_us_per_token}-$a->{die0_timeline}{device_us_per_token},
          timeline_delta_us_per_token=>{map {$_=>$b->{die0_timeline}{components_us_per_token}{$_}-$a->{die0_timeline}{components_us_per_token}{$_}} @components},
          distinct_attention_work_delta_us=>{map {$_=>$b->{distinct_attention_layer_work_us}{$_}-$a->{distinct_attention_layer_work_us}{$_}} @attention}};
    }
    return \%w;
}
sub validate_data {
    my($traced,$untraced,$phase)=@_;
    my($runs,$ts)=validate_runs($traced,1);my($uruns,$us)=validate_runs($untraced,0);
    my $ps=validate_phases($phase,$runs,$traced->{manifest}{configs});
    return {traced_summary=>$ts,untraced_summary=>$us,matched_phase_deltas=>witness($runs),
      traced_runs=>0+keys(%$runs),untraced_runs=>0+keys(%$uruns),%$ps};
}
sub replay {
    my($root,$out)=@_;
    die "output directory already exists\n" if -e $out || -l $out;
    my $ids=verify_identities($root);
    my $result=validate_data(map {read_json("$root/$_")} @required[0..2]);
    my %outputs=('traced-summary.json'=>$result->{traced_summary},
      'untraced-summary.json'=>$result->{untraced_summary},
      'matched-phase-deltas.json'=>$result->{matched_phase_deltas});
    my $max=0;
    for my $p(sort keys %outputs){my $e=compare($outputs{$p},read_json("$root/$p"),$p);$max=$e if $e>$max;}
    my $record={status=>'PASS',scope=>'Compact archive consistency and numerical replay only; raw phase CSVs were not re-read; no GPU inference or issue-acceptance claim.',
      traced_runs=>$result->{traced_runs},untraced_runs=>$result->{untraced_runs},
      die_frames=>$result->{die_frames},max_recomputed_residual_us=>$result->{max_recomputed_residual_us},
      max_summary_difference=>$max,summary_tolerance=>1e-8,%$ids,
      analyzer_sha256=>{'kern/DecodeArchive.pm'=>DecodeForensics::sha(__FILE__),
        'kern/decode_forensics.pl'=>DecodeForensics::sha(dirname(__FILE__).'/decode_forensics.pl')}};
    File::Path::make_path($out);
    DecodeForensics::save("$out/$_",$outputs{$_}) for sort keys %outputs;
    DecodeForensics::save("$out/replay.json",$record);
    DecodeForensics::output_identities($out);
    print JSON::PP->new->canonical->pretty->encode($record);
    return 0;
}
1;
