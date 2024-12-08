import streamlit as st
import unittest
import sys
import io
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import logging
import traceback
from pathlib import Path

# Custom test result class to capture test data
class StreamlitTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.test_results: List[Dict[str, Any]] = []
        self.current_test: Dict[str, Any] = {}
        self.start_time = datetime.now()

    def startTest(self, test):
        self.current_test = {
            'name': test.id(),
            'status': 'running',
            'start_time': datetime.now(),
            'logs': [],
            'error': None
        }
        self.test_results.append(self.current_test)

    def addSuccess(self, test):
        self.current_test['status'] = 'passed'
        self.current_test['end_time'] = datetime.now()
        self.current_test['duration'] = (self.current_test['end_time'] - self.current_test['start_time']).total_seconds()

    def addError(self, test, err):
        self.current_test['status'] = 'error'
        self.current_test['end_time'] = datetime.now()
        self.current_test['duration'] = (self.current_test['end_time'] - self.current_test['start_time']).total_seconds()
        self.current_test['error'] = {
            'type': err[0].__name__,
            'message': str(err[1]),
            'traceback': traceback.format_exception(*err)
        }

    def addFailure(self, test, err):
        self.current_test['status'] = 'failed'
        self.current_test['end_time'] = datetime.now()
        self.current_test['duration'] = (self.current_test['end_time'] - self.current_test['start_time']).total_seconds()
        self.current_test['error'] = {
            'type': err[0].__name__,
            'message': str(err[1]),
            'traceback': traceback.format_exception(*err)
        }

    def addSkip(self, test, reason):
        self.current_test['status'] = 'skipped'
        self.current_test['end_time'] = datetime.now()
        self.current_test['duration'] = (self.current_test['end_time'] - self.current_test['start_time']).total_seconds()
        self.current_test['skip_reason'] = reason

class TestMonitor:
    def __init__(self):
        self.test_results = []
        self.overall_status = 'not_started'
        self.start_time = None
        self.end_time = None

    def run_tests(self):
        """Run the test suite and capture results"""
        self.start_time = datetime.now()
        self.overall_status = 'running'
        
        # Load test suite
        loader = unittest.TestLoader()
        suite = loader.discover(str(Path(__file__).parent))
        
        # Run tests with custom result handler
        result = StreamlitTestResult()
        suite.run(result)
        
        self.test_results = result.test_results
        self.end_time = datetime.now()
        self.overall_status = 'completed'
        
        return self.test_results

def main():
    st.set_page_config(
        page_title="Test Monitor",
        page_icon="🧪",
        layout="wide"
    )

    st.title("🧪 Test Monitor Dashboard")

    # Initialize session state
    if 'monitor' not in st.session_state:
        st.session_state.monitor = TestMonitor()
        st.session_state.auto_refresh = False
        st.session_state.last_refresh = None

    # Control panel
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        if st.button("Run Tests", key="run_tests"):
            st.session_state.monitor = TestMonitor()
            st.session_state.monitor.run_tests()
            st.session_state.last_refresh = datetime.now()
    
    with col2:
        st.session_state.auto_refresh = st.checkbox(
            "Auto Refresh (5s)", 
            value=st.session_state.auto_refresh
        )
    
    with col3:
        if st.session_state.last_refresh:
            st.write(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    # Auto refresh logic
    if st.session_state.auto_refresh:
        time.sleep(5)
        st.rerun()

    # Overall status
    st.header("Overall Status")
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        total_tests = len(st.session_state.monitor.test_results)
        passed_tests = len([t for t in st.session_state.monitor.test_results if t['status'] == 'passed'])
        failed_tests = len([t for t in st.session_state.monitor.test_results if t['status'] in ['failed', 'error']])
        
        st.metric("Total Tests", total_tests)
    
    with status_col2:
        st.metric("Passed Tests", passed_tests, delta=f"{passed_tests/total_tests*100:.1f}%" if total_tests else None)
    
    with status_col3:
        st.metric("Failed Tests", failed_tests, delta=f"{failed_tests/total_tests*100:.1f}%" if total_tests else None)

    # Test Results
    st.header("Test Results")
    
    for test in st.session_state.monitor.test_results:
        with st.expander(f"{test['name']} - {test['status'].upper()}", expanded=test['status'] in ['failed', 'error']):
            col1, col2 = st.columns([3,1])
            
            with col1:
                st.write(f"**Status:** {test['status'].upper()}")
                if 'duration' in test:
                    st.write(f"**Duration:** {test['duration']:.2f}s")
                
                if test.get('error'):
                    st.error(f"**Error Type:** {test['error']['type']}")
                    st.error(f"**Error Message:** {test['error']['message']}")
                    with st.expander("Show Traceback"):
                        st.code(''.join(test['error']['traceback']), language='python')
            
            with col2:
                # Status indicator
                color = {
                    'passed': 'green',
                    'failed': 'red',
                    'error': 'red',
                    'skipped': 'gray',
                    'running': 'blue'
                }.get(test['status'], 'gray')
                
                st.markdown(f"""
                    <div style="
                        width: 50px;
                        height: 50px;
                        border-radius: 25px;
                        background-color: {color};
                        margin: auto;
                    "></div>
                """, unsafe_allow_html=True)

    # Progress bar
    if st.session_state.monitor.overall_status == 'running':
        progress = len([t for t in st.session_state.monitor.test_results if t['status'] != 'running']) / total_tests
        st.progress(progress)

if __name__ == "__main__":
    main() 