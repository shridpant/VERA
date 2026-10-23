#!/usr/bin/env python3
"""Categorized tests with persistent logs; explicit real-model acceptance suite."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SUITES={
 'configuration':'tests.configuration',
 'datasets':'tests.datasets',
 'preprocessing':'tests.preprocessing',
 'backends':'tests.backends',
 'interventions':'tests.interventions',
 'parsing':'tests.parsing',
 'refinement':'tests.refinement',
 'integration':'tests.integration',
 'runtime':'tests.runtime',
 'real-model':'tests.real_model',
}

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
    def flush(self):
        for stream in self.streams:stream.flush()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['all',*SUITES],default='all');args=ap.parse_args()
    if args.suite=='real-model':os.environ['VERA_TEST_REAL_MODEL']='1'
    else:os.environ.pop('VERA_TEST_REAL_MODEL',None)
    start=datetime.datetime.now(datetime.timezone.utc)
    report=ROOT/'tests/reports'/args.suite/start.strftime('%Y%m%dT%H%M%S%fZ');report.mkdir(parents=True)
    loader=unittest.TestLoader();suite=unittest.TestSuite()
    names=SUITES if args.suite=='all' else [args.suite]
    for name in names:suite.addTests(loader.discover(str(ROOT/SUITES[name].replace('.','/')),pattern='test_*.py',top_level_dir=str(ROOT)))
    with (report/'tests.log').open('w') as log:
        result=unittest.TextTestRunner(stream=Tee(sys.stdout,log),verbosity=2).run(suite)
    summary={'suite':args.suite,'started_at':start.isoformat(),'python':sys.executable,
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'skipped':len(result.skipped),'successful':result.wasSuccessful(),
             'real_model_requested':args.suite=='real-model',
             'real_model_passed':args.suite=='real-model' and result.wasSuccessful() and result.testsRun>0 and not result.skipped}
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Test report:',report)
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':sys.exit(main())
