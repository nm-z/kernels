// lfm3-data TOKENIZER.json OUTDIR CTX < list of .jsonl[.zst] paths: what lfm3-1.7b.jl --train reads, for a model that says what the user
// would say next. Each transcript's user and assistant text becomes one token array, rendered as the chat template does; a typed
// reply after the assistant's text is an example; every example is offsets into its transcript's array, nothing is copied.
//   tokens.bin   Int32, the transcripts' arrays one after another
//   trans.bin    (first token u64, tokens u32, date yyyymmdd u32, held out u32, 0) a transcript
//   examples.bin (transcript, context start, start, end) u32, offsets in the transcript: a typed stretch of a reply (a paste in the
//                reply splits it: the paste is context); the reply's <|im_end|> is in its last stretch
//   names.txt    a transcript's source path, a line
use rayon::prelude::*;
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use tokenizers::Tokenizer;

const BOS: i32 = 124894;
const IM_START: i32 = 124899;
const IM_END: i32 = 124900;
const MAX_REPLY: usize = 512; // tokens; past this it's a paste, not a reply

#[derive(Clone, PartialEq)]
enum Role { User, Assistant }
struct Turn { role: Role, text: String, target: bool }
struct Trans { name: String, date: u32, turns: Vec<Turn> }

fn ctext(c: Option<&Value>) -> Option<String> {
    match c {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(a)) => {
            let mut o = String::new();
            for b in a {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | Some("output_text") => o.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some("image") | Some("input_image") => {}
                    _ => return None, // a tool result: not typed
                }
            }
            Some(o)
        }
        _ => None,
    }
}
const PASTE: [char; 2] = ['\u{e000}', '\u{e001}']; // around a paste in a turn's text: read as context, never a target (the user didn't write it)
// what the harness adds to a typed message goes; a paste stays, between the marks
fn clean(m: &str) -> String {
    let m = m.replace(PASTE, "");
    let mut it = m.split("<pasted_content");
    let mut s = it.next().unwrap().to_string();
    for p in it {
        let p = p.split_once('>').map_or("", |x| x.1); // past the opening tag
        let (body, rest) = p.split_once("</pasted_content").map_or((p, ""), |(b, r)| (b, r.split_once('>').map_or("", |x| x.1)));
        s.push(PASTE[0]);
        s.push_str(body.trim());
        s.push(PASTE[1]);
        s.push_str(rest);
    }
    for t in ["<widget_context_hint", "<system-reminder>", "<ide_selection>", "<ide_opened_file>"] {
        if let Some(a) = s.find(t) {
            s.truncate(a);
        }
    }
    s.replace("<|", "< |").trim().to_string()
}
fn add(tr: &mut Vec<Turn>, role: Role, text: String, target: bool) {
    if text.is_empty() {
        return;
    }
    if role == Role::Assistant {
        if let Some(l) = tr.last_mut() {
            if l.role == Role::Assistant {
                l.text.push_str("\n\n");
                l.text.push_str(&text);
                return;
            }
        }
    }
    tr.push(Turn { role, text, target });
}
fn date(l: &Value) -> Option<u32> {
    let t = l.get("timestamp").and_then(Value::as_str)?;
    (t.len() >= 10).then(|| t[..10].replace('-', "").parse().ok()).flatten()
}
fn parse(f: &str) -> Option<Trans> {
    let cx = f.contains("/.codex/") || f.contains("/codex-sessions/");
    if f.contains("/subagents/") {
        return None;
    }
    let h = std::fs::File::open(f).ok()?;
    let r: Box<dyn BufRead> = if f.ends_with(".zst") { Box::new(BufReader::with_capacity(1 << 20, zstd::Decoder::new(h).ok()?)) } else { Box::new(BufReader::with_capacity(1 << 20, h)) };
    let (mut turns, mut d, mut first, mut human) = (Vec::new(), 0u32, true, true);
    for line in r.split(b'\n') {
        let Ok(line) = line else { break };
        let Ok(l) = serde_json::from_slice::<Value>(&line) else { continue };
        if d == 0 {
            d = date(&l).unwrap_or(0);
        }
        let ty = l.get("type").and_then(Value::as_str).unwrap_or("");
        if cx {
            let Some(p) = l.get("payload") else { continue };
            if ty == "session_meta" {
                let o = p.get("originator").and_then(Value::as_str).unwrap_or("");
                human = !matches!(p.get("source"), Some(Value::Object(_))) && (o == "Codex Desktop" || o == "codex-tui");
                continue;
            }
            if ty != "response_item" || p.get("type").and_then(Value::as_str) != Some("message") {
                continue;
            }
            let Some(t) = ctext(p.get("content")) else { continue };
            match p.get("role").and_then(Value::as_str) {
                Some("assistant") => add(&mut turns, Role::Assistant, t.replace("<|", "< |").trim().to_string(), false),
                Some("user") => {
                    let t = t.trim();
                    if t.starts_with("# AGENTS.md") || t.starts_with("From ") || (t.starts_with('<') && !t.starts_with("<pasted_content")) {
                        continue;
                    }
                    let c = clean(t);
                    let ok = human && !first && matches!(turns.last(), Some(Turn { role: Role::Assistant, .. }));
                    first = false;
                    add(&mut turns, Role::User, c, ok);
                }
                _ => {}
            }
        } else {
            if l.get("isSidechain").and_then(Value::as_bool) == Some(true) || l.get("isMeta").and_then(Value::as_bool) == Some(true) {
                continue;
            }
            let content = l.get("message").and_then(|m| m.get("content"));
            if ty == "assistant" {
                let mut o = String::new();
                for b in content.and_then(Value::as_array).into_iter().flatten() {
                    if b.get("type").and_then(Value::as_str) == Some("text") {
                        o.push_str(b.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                }
                add(&mut turns, Role::Assistant, o.replace("<|", "< |").trim().to_string(), false);
            } else if ty == "user" {
                let Some(t) = ctext(content) else { continue };
                let t = t.trim();
                if t.starts_with("From ") || t.starts_with("This session is being continued") || (t.starts_with('<') && !t.starts_with("<pasted_content")) {
                    continue;
                }
                let k = l.get("origin").and_then(|o| o.get("kind")).and_then(Value::as_str);
                let c = clean(t);
                let ok = k == Some("human") && !first && matches!(turns.last(), Some(Turn { role: Role::Assistant, .. }));
                first = false;
                add(&mut turns, Role::User, c, ok);
            }
        }
    }
    turns.iter().any(|t| t.target).then(|| Trans { name: f.to_string(), date: d, turns })
}
fn phash(a: u64, t: &Turn) -> u64 {
    let mut s = DefaultHasher::new();
    a.hash(&mut s);
    (t.role == Role::User).hash(&mut s);
    t.text.hash(&mut s);
    s.finish()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let tok = Tokenizer::from_file(&a[1]).unwrap();
    let out = std::path::Path::new(&a[2]);
    let ctx: usize = a[3].parse().unwrap();
    std::fs::create_dir_all(out).unwrap();
    let files: Vec<String> = std::io::stdin().lock().lines().map(Result::unwrap).collect();
    let mut ts: Vec<Trans> = files.par_iter().filter_map(|f| parse(f)).collect();
    let nt0: usize = ts.iter().map(|t| t.turns.iter().filter(|u| u.target).count()).sum();
    // a resumed or forked session repeats a prefix of another: its copies aren't targets. The longest transcripts go first, so the
    // original's replies stay; a transcript left with no target goes.
    ts.sort_by(|x, y| y.turns.len().cmp(&x.turns.len()).then(x.name.cmp(&y.name)));
    let mut seen = HashSet::new();
    for t in ts.iter_mut() {
        let mut h = 0u64;
        for u in t.turns.iter_mut() {
            h = phash(h, u);
            if !seen.insert(h) {
                u.target = false;
            }
        }
    }
    ts.retain(|t| t.turns.iter().any(|u| u.target));
    ts.sort_by(|x, y| x.date.cmp(&y.date).then(x.name.cmp(&y.name)));
    // 20% of the transcripts of each date, in time order within it: every fifth, not at random
    let mut hold = vec![false; ts.len()];
    let (mut i, mut k) = (0, 0);
    while i < ts.len() {
        k = if i > 0 && ts[i].date == ts[i - 1].date { k + 1 } else { 0 };
        hold[i] = k % 5 == 2;
        i += 1;
    }
    let enc = |s: &str| -> Vec<i32> { tok.encode_fast(s, false).unwrap().get_ids().iter().map(|&x| x as i32).collect() };
    let (us, asn) = (enc("user\n"), enc("assistant\n"));
    let nl = enc("\n");
    let nrep = std::sync::atomic::AtomicU64::new(0);
    let rendered: Vec<(Vec<i32>, Vec<(u32, u32, u32)>)> = ts
        .par_iter()
        .map(|t| {
            let mut v = vec![BOS];
            let mut starts: Vec<u32> = vec![];
            let mut ex = vec![];
            for u in &t.turns {
                let s = v.len();
                starts.push(s as u32);
                v.push(IM_START);
                v.extend_from_slice(if u.role == Role::User { &us } else { &asn });
                // the typed stretches of the turn (the last one has its <|im_end|>): a paste between them is tokens, not a target
                let mut typed: Vec<(usize, usize)> = vec![];
                for (k, piece) in u.text.split(PASTE).enumerate() {
                    let a = v.len();
                    v.extend(enc(piece));
                    if k % 2 == 0 && v.len() > a {
                        typed.push((a, v.len()));
                    }
                }
                match typed.last_mut() {
                    Some(l) if l.1 == v.len() => l.1 += 1,
                    _ => typed.push((v.len(), v.len() + 1)),
                }
                v.push(IM_END);
                let n: usize = typed.iter().map(|t| t.1 - t.0).sum();
                if u.target && n <= MAX_REPLY && n >= 2 {
                    for &(rs, re) in &typed {
                        // the context starts at the first turn inside the limit, or mid-turn when what's before the stretch is longer than it
                        let lo = rs.saturating_sub(ctx) as u32;
                        let cs = starts.iter().copied().skip_while(|&q| q < lo).next().filter(|&q| (q as usize) < s).unwrap_or(lo);
                        ex.push((cs, rs as u32, re as u32));
                    }
                    nrep.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                v.extend_from_slice(&nl);
            }
            (v, ex)
        })
        .collect();
    let mut ft = BufWriter::new(std::fs::File::create(out.join("tokens.bin")).unwrap());
    let mut fr = BufWriter::new(std::fs::File::create(out.join("trans.bin")).unwrap());
    let mut fe = BufWriter::new(std::fs::File::create(out.join("examples.bin")).unwrap());
    let mut fn_ = BufWriter::new(std::fs::File::create(out.join("names.txt")).unwrap());
    let (mut off, mut nw, mut nex, mut nho, mut rtok, mut ctok) = (0u64, 0u32, 0u64, 0u64, 0u64, 0u64);
    let mut rl = vec![];
    for (i, (v, ex)) in rendered.iter().enumerate() {
        if ex.is_empty() {
            continue;
        }
        for x in v {
            ft.write_all(&x.to_le_bytes()).unwrap();
        }
        fr.write_all(&off.to_le_bytes()).unwrap();
        for w in [v.len() as u32, ts[i].date, hold[i] as u32, 0u32] {
            fr.write_all(&w.to_le_bytes()).unwrap();
        }
        writeln!(fn_, "{}", ts[i].name).unwrap();
        for e in ex {
            for w in [nw, e.0, e.1, e.2] {
                fe.write_all(&w.to_le_bytes()).unwrap();
            }
            nex += 1;
            nho += hold[i] as u64;
            rtok += (e.2 - e.1) as u64;
            ctok += (e.1 - e.0) as u64;
            rl.push(e.2 - e.1);
        }
        off += v.len() as u64;
        nw += 1;
    }
    rl.sort();
    println!("transcripts read {}, with targets {}, written {}", files.len(), ts.len(), nw);
    println!("targets before removing copies {nt0}; replies {}, their typed stretches (examples) {nex}, held out {nho}", nrep.into_inner());
    println!("tokens in arrays {off}; reply tokens {rtok}, context tokens as if each were separate {ctok}");
    println!("reply median {}, p90 {}, max {}", rl[rl.len() / 2], rl[rl.len() * 9 / 10], rl[rl.len() - 1]);
}
