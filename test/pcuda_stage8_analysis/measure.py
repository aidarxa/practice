#!/usr/bin/env python3
"""Collect complete SQL sessions first; validate/oracle/analyze only after all collection ends."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'test/pcuda_stage0'))
from native_timing import timed_command,read_time,trace_summary
from sql_baseline import run_session,REFERENCE_GROUPS,cli_rows,reference_rows


def loader_command(loader,binary,probe):
    return [str(loader),*(['--preload',str(probe)] if probe else []),str(binary)]


def interval_union_ns(intervals):
    total=0;previous_end=0
    for begin,end in sorted(intervals):
        total+=max(0,end-max(begin,previous_end))
        previous_end=max(previous_end,end)
    return total


def module_waves(text):
    waves=[]
    for line in text.splitlines():
        if not line.startswith('Module wave:'):continue
        match=re.fullmatch(r'Module wave: first=(\d+) count=(\d+) begin_ns=(\d+) end_ns=(\d+)',line)
        if not match:raise ValueError('Malformed native build wave boundary')
        waves.append(dict(zip(('first','count','begin_ns','end_ns'),map(int,match.groups()))))
    for wave in waves:
        if set(wave)!={'first','count','begin_ns','end_ns'} or not 1<=wave['count']<=4 or wave['end_ns']<wave['begin_ns']:
            raise ValueError('Invalid native build wave observation')
    return waves


def validate_probe(data):
    if data.get('clock')!='steady_ns' or data.get('dropped')!=0 or data.get('event_count')!=len(data.get('events',[])):
        raise ValueError('Incomplete or invalid observation stream')
    events=data['events']
    full={event['compilation']:event for event in events if event['kind']=='full_transformation'}
    if not full or 0 in full or len(full)!=data.get('compilations') or len(full)!=sum(event['kind']=='full_transformation' for event in events):
        raise ValueError('Missing or duplicate compilation observations')
    for group in full:
        kinds={event['kind'] for event in events if event['compilation']==group}
        if not {'optimizer','external_tool','hiprtc_complete'}<=kinds:
            raise ValueError('An expected nested hook did not run; absence is not zero cost')
    totals={}
    for event in events:
        if event['status']!=0 or event['end_ns']<event['begin_ns']:
            raise ValueError('Failed or inverted native observation')
        group=event['compilation']
        if group:
            parent=full.get(group)
            if not parent or event['begin_ns']<parent['begin_ns'] or event['end_ns']>parent['end_ns']:
                raise ValueError('Event lies outside its fullTransformation')
        totals[event['kind']+'_ms']=totals.get(event['kind']+'_ms',0)+(event['end_ns']-event['begin_ns'])/1e6
    return dict(compilations=len(full),**totals)


def imported(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def collect(args):
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='s2-series-',dir=args.output_root.resolve()))
    print(output,flush=True)
    queries=args.query or ['q11','q21','q42']
    modes=args.mode or ['stock','probe']
    schedules=args.schedule or ['barrier']
    diagnostics=[False,True] if args.diagnostics=='both' else [args.diagnostics=='on']
    records=[]
    for session in range(args.sessions):
        rotated=queries[session%len(queries):]+queries[:session%len(queries)]
        for query in rotated:
            for traced in diagnostics if session%2==0 else diagnostics[::-1]:
                variants=[(schedule,mode) for schedule in (schedules if session%2==0 else schedules[::-1])
                          for mode in (modes if (session+int(traced))%2==0 else modes[::-1])]
                for schedule,mode in variants:
                    suffix='-overlap' if schedule=='overlap' else ''
                    directory=output/f'{session}-{query}-{mode}-{"traced" if traced else "plain"}{suffix}'
                    directory.mkdir()
                    config=directory/'crystal.conf'
                    config.write_text('dump_generated_code=off\nextended_timing=on\noutput_limit=off\nmemory_guard_enabled=on\nreuse_scratch_buffers=true\njit_keep_artifacts=on\njit_jobs=4\n'+f'jit_artifact_dir={directory}\njit_traces={"on" if traced else "off"}\n')
                    if args.schedule:
                        with config.open('a') as file:file.write(f'jit_overlap_experiment={"on" if schedule=="overlap" else "off"}\n')
                    sql=(ROOT/f'test/queries/sql/{query}.sql').read_text().strip()
                    lines=['\\dump off','\\timing on','\\limit all']
                    for index in range(args.runs):lines += [f'\\echo __S2_{index}__',sql]
                    lines += [f'\\echo __S2_{args.runs}__','\\q']
                    stdin='\n'.join(lines)+'\n';(directory/'stdin.sql').write_text(stdin)
                    environment={key:value for key,value in os.environ.items() if key!='LD_PRELOAD' and not key.startswith('CRYSTAL_S2_')}
                    environment.update(LC_ALL='C',TMPDIR='/tmp/opencode',ACPP_ADAPTIVITY_LEVEL='1',ACPP_ALLOCATION_TRACKING='1',ACPP_APPDB_DIR=str(directory/'device-cache'),CRYSTAL_CONFIG=str(config))
                    environment['CRYSTAL_JIT_OVERLAP_EXPERIMENT']='on' if schedule=='overlap' else 'off'
                    if mode=='probe':environment.update(CRYSTAL_S2_EVENTS=str(directory/'s2-events.json'),CRYSTAL_S2_DIRECTORY=str(directory),CRYSTAL_S2_NATIVE_TIMER=str(ROOT/'test/pcuda_jit_tiering/native_tool.sh'))
                    command=loader_command(args.loader.resolve(),args.build.resolve()/'db_cli',args.probe.resolve() if mode=='probe' else None)
                    argv=timed_command([args.build.resolve()/'pcuda_acceptance_first_result',directory/'first-result.json',*command],directory/'process.time.json',directory/'process.stderr')
                    entry=dict(session=session,query=query,mode=mode,schedule=schedule,timing_model='pcuda-modules-overlap-exp-v1' if schedule=='overlap' else args.timing_model,
                               traces=traced,directory=directory.name,argv=argv,environment={key:value for key,value in environment.items() if key.startswith(('ACPP_','CRYSTAL_','HIP_','HSA_')) or key in ('LC_ALL','TMPDIR')})
                    (directory/'command.json').write_text(json.dumps(entry,indent=2)+'\n')
                    rc,stdout,stderr,timeout=run_session(argv,stdin,ROOT,environment,600)
                    (directory/'stdout.txt').write_text(stdout);(directory/'timer.stderr').write_text(stderr)
                    entry.update(returncode=rc,timeout=timeout);records.append(entry)
                    (output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
                    print(directory.name,rc,flush=True)
                    if rc or timeout:raise RuntimeError('Session failed: '+str(directory))
    report=dict(order=records,sessions=args.sessions,runs=args.runs,queries=queries,modes=modes,schedules=schedules,diagnostics=args.diagnostics,timing_model=args.timing_model,
                binary_sha256=hashlib.sha256((args.build/'db_cli').read_bytes()).hexdigest(),probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
                boundary='Full db_cli SQL; direct loader for matched stock/probe modes; --preload does not propagate into acpp children',
                caches='Fresh process/module/appdb each session; retained within warm; OS/HIP/driver caches not cleared; no online analysis')
    (output/'series.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


def analyze(args):
    series=json.loads((args.series/'series.json').read_text())
    runner=imported('s2_metrics','olap_bench/tools/run_olap_bench.py')
    acceptance=imported('s2_acceptance','test/pcuda_stage8/measure.py')
    distribution=imported('s2_distribution','test/pcuda_stage5/analyze_series.py').distribution
    oracle=json.loads(args.q11_oracle.read_text())
    quality=imported('s2_diagnostics','test/pcuda_stage7/validate.py')
    if oracle['sql_q11']!=oracle['year_1993']+oracle['january_1_1994']:
        raise ValueError('Invalid independent Q1.1 boundary oracle')
    sessions=[]
    for entry in series['order']:
        directory=args.series/entry['directory'];stdout=(directory/'stdout.txt').read_text()
        query=entry['query'];reference=reference_rows((ROOT/f'test/queries/result/scale10/{query[1:]}.txt').read_text())
        diagnostics=quality.unexpected_diagnostics((directory/'process.stderr').read_text()+(directory/'timer.stderr').read_text())
        if entry['returncode'] or entry['timeout'] or diagnostics:raise ValueError('Failed session or unexpected diagnostics: '+str(directory))
        if query=='q11' and reference!=Counter({(oracle['year_1993'],):1}):raise ValueError('Q1.1 file oracle changed')
        rows=[]
        for index in range(series['runs']):
            if any(stdout.count(f'__S2_{n}__\n')!=1 for n in (index,index+1)):raise ValueError('Missing or repeated query marker')
            text=stdout.split(f'__S2_{index}__\n',1)[1].split(f'__S2_{index+1}__\n',1)[0]
            if runner.has_sql_error(text):raise ValueError('SQL error')
            metrics=runner.parse_metrics(text);acceptance.check_record(metrics,index==0,entry.get('timing_model','pcuda-modules-v1'))
            actual=cli_rows(text,REFERENCE_GROUPS[query]);file_matches=actual==reference
            cpu_matches=actual==Counter({(oracle['sql_q11'],):1}) if query=='q11' else None
            if not (cpu_matches if query=='q11' else file_matches):raise ValueError('Independent result mismatch')
            waves=module_waves(text)
            if entry.get('schedule')=='overlap' and sum(wave['count'] for wave in waves)!=metrics['module_compiler_calls']:
                raise ValueError('Native wave observations do not cover the actual compiler events')
            rows.append(dict(index=index,phase='cold' if index==0 else 'warm',metrics=metrics,waves=waves,file_matches=file_matches,q11_cpu_oracle_matches=cpu_matches))
        result=dict(**entry,records=rows,native_process=read_time(directory/'process.time.json'),first_result=json.loads((directory/'first-result.json').read_text()))
        result['traces_ms']={str(path.relative_to(directory)):trace_summary(path) for path in directory.glob('db-pcuda-*/*/*.json') if path.name.endswith(('.clang.json','.lld.json'))}
        if entry['mode']=='probe':
            events=json.loads((directory/'s2-events.json').read_text());result['probe']=validate_probe(events)
            result['full_compilations']=[event for event in events['events'] if event['kind']=='full_transformation']
            result['wave_s2_overlap']=[dict(wave=wave,full_transformation_overlap_ms=interval_union_ns([
                (max(wave['begin_ns'],event['begin_ns']),min(wave['end_ns'],event['end_ns']))
                for event in result['full_compilations'] if event['begin_ns']<wave['end_ns'] and event['end_ns']>wave['begin_ns']])/1e6)
                for wave in rows[0]['waves']]
            result['hook_counts']=dict(Counter(event['kind'] for event in events['events']))
            result['native_s2_tools']=[dict(event=event,native=read_time(event['name']+'.time.json')) for event in events['events'] if event['kind']=='external_tool']
        sessions.append(result)
    aggregates=[]
    for query in series['queries']:
        variants=[(mode,schedule) for mode in series['modes'] for schedule in series.get('schedules',['barrier'])]
        for mode,schedule in variants:
            for traced in (False,True):
                group=[session for session in sessions if session['query']==query and session['mode']==mode and session['traces']==traced and session.get('schedule','barrier')==schedule]
                if not group:continue
                fields=tuple(dict.fromkeys(('total_query_ms','generated_execute_ms','host_fetch_ms','library_load_ms','sql_to_pcuda_ms',*runner.MODULE_TIMING_FIELDS)))
                item=dict(query=query,mode=mode,schedule=schedule,traces=traced,cold={field:distribution([session['records'][0]['metrics'][field] for session in group]) for field in fields},
                          warm={field:distribution([record['metrics'][field] for session in group for record in session['records'][1:]]) for field in fields})
                item['native_process']={field:distribution([session['native_process'][field] for session in group]) for field in ('wall_ms','user_ms','system_ms')}
                item['spawn_to_first_ms']=distribution([session['first_result']['process_start_to_first_result_ms'] for session in group])
                item['startup']={field:distribution([session['records'][0]['metrics'][field] for session in group]) for field in runner.OBSERVATION_TIMING_FIELDS if field!='sql_to_pcuda_ms'}
                if mode=='probe':
                    kinds=set.intersection(*(set(session['probe']) for session in group))
                    item['probe']={kind:distribution([session['probe'][kind] for session in group]) for kind in sorted(kinds)}
                aggregates.append(item)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(success=True,independent_oracles_verified=True,full_file_parity=all(record['file_matches'] for session in sessions for record in session['records']),
                sessions=sessions,aggregate=aggregates,cpu_results=sum(len(session['records']) for session in sessions),
                file_matches=sum(record['file_matches'] for session in sessions for record in session['records']),
                known_q11_file_mismatches=sum(not record['file_matches'] for session in sessions for record in session['records']),
                note='S2 child intervals are nested; HIP runtime totals are PROCESS scope (loading, cold, warm, teardown), not query-only; no pure-JIT subtraction or SLA acceptance implied')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for item in aggregates:
        if not item['traces']:print(item['query'],item['mode'],item['schedule'],'summary',item['cold']['total_query_ms']['median'],'build',item['cold']['build_elapsed_ms']['median'],'S2',item.get('probe',{}).get('full_transformation_ms'))
    print('Verified CPU results:',report['cpu_results'],'known Q1.1 file mismatches:',report['known_q11_file_mismatches'])
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    collect_parser=sub.add_parser('collect')
    collect_parser.add_argument('--build',type=Path,required=True);collect_parser.add_argument('--probe',type=Path,required=True)
    collect_parser.add_argument('--loader',type=Path,required=True);collect_parser.add_argument('--output-root',type=Path,required=True)
    collect_parser.add_argument('--query',choices=('q11','q21','q42'),action='append');collect_parser.add_argument('--mode',choices=('stock','probe'),action='append')
    collect_parser.add_argument('--schedule',choices=('barrier','overlap'),action='append',help='Explicit opt-in to the experimental scheduling/timing contract')
    collect_parser.add_argument('--sessions',type=int,default=7);collect_parser.add_argument('--runs',type=int,default=9)
    collect_parser.add_argument('--diagnostics',choices=('both','on','off'),default='both')
    collect_parser.add_argument('--timing-model',choices=('pcuda-modules-v1',),required=True)
    analyze_parser=sub.add_parser('analyze');analyze_parser.add_argument('--series',type=Path,required=True)
    analyze_parser.add_argument('--output',type=Path,required=True);analyze_parser.add_argument('--q11-oracle',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='collect' and (args.sessions<1 or args.runs<2):parser.error('Require positive sessions and at least one warm repeat')
    return collect(args) if args.action=='collect' else analyze(args)


if __name__=='__main__':raise SystemExit(main())
