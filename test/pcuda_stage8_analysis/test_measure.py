"""Behavioral checks for the diagnostic reader, never a performance timer."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from measure import validate_probe, loader_command, imported, module_waves, interval_union_ns


class ProbeReader(unittest.TestCase):
    def event(self,kind,begin=10,end=20,compilation=1):
        return dict(kind=kind,begin_ns=begin,end_ns=end,compilation=compilation,status=0,name='kernel',bytes=0,arguments=[])

    def valid(self):
        events=[self.event('full_transformation',10,100),self.event('optimizer',20,30),self.event('hiprtc_complete',40,80),self.event('external_tool',30,40)]
        return dict(clock='steady_ns',pid=1,dropped=0,event_count=len(events),compilations=1,events=events)

    def test_nested_intervals_not_double_counted(self):
        result=validate_probe(self.valid())
        self.assertEqual(result['full_transformation_ms'],0.00009)
        self.assertEqual(result['optimizer_ms'],0.00001)

    def test_reject_loss_or_bad_status(self):
        for field,value in [('dropped',1),('event_count',0),('compilations',0),('clock','wall')]:
            data=self.valid();data[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_probe(data)
        data=self.valid();data['events'][0]['status']=1
        with self.assertRaises(ValueError):validate_probe(data)

    def test_reject_inverted_and_outside_child(self):
        for index,field,value in [(0,'end_ns',1),(1,'end_ns',101),(1,'compilation',2)]:
            data=self.valid();data['events'][index][field]=value
            with self.subTest(index=index,field=field),self.assertRaises(ValueError):validate_probe(data)

    def test_noninherited_preload_is_explicit(self):
        self.assertEqual(loader_command(Path('/loader'),Path('/db_cli'),None),['/loader','/db_cli'])
        self.assertEqual(loader_command(Path('/loader'),Path('/db_cli'),Path('/probe.so')),['/loader','--preload','/probe.so','/db_cli'])

    def test_missing_inner_hook_is_not_zero_cost(self):
        for kind in ('optimizer','external_tool','hiprtc_complete'):
            data=self.valid();data['events']=[event for event in data['events'] if event['kind']!=kind]
            data['event_count']=len(data['events'])
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_probe(data)

    def test_overlap_model_cannot_publish_additive_component_sum(self):
        runner=imported('overlap_timing_test','olap_bench/tools/run_olap_bench.py')
        result=runner.normalize_timing_record(dict(reported_timing_model='pcuda-modules-overlap-exp-v1',
            prepare_ms=1,codegen_ms=2,compile_ms=100,library_load_ms=1,generated_execute_ms=200,host_fetch_ms=1,total_query_ms=250))
        self.assertIsNone(result['timing_component_sum_ms'])
        self.assertIsNone(result['engine_component_sum_ms'])

    def test_wave_boundaries_are_integral_and_bounded(self):
        self.assertEqual(module_waves('Module wave: first=4 count=1 begin_ns=100 end_ns=200\n'),
                         [dict(first=4,count=1,begin_ns=100,end_ns=200)])
        for line in ('Module wave: first=0 count=5 begin_ns=1 end_ns=2',
                     'Module wave: first=0 count=1 begin_ns=2 end_ns=1','Module wave: first=0 count=1',
                     'Module wave: first=0 count=1 begin_ns=1.5 end_ns=2'):
            with self.subTest(line=line),self.assertRaises(ValueError):module_waves(line)

    def test_overlap_union_does_not_add_nested_compilations(self):
        self.assertEqual(interval_union_ns([(20,80),(30,40),(80,90),(95,100)]),75)
        self.assertEqual(interval_union_ns([]),0)


if __name__=='__main__':unittest.main()
