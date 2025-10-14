import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from my_investment.core import hello_world
def test_hello_world():
    assert hello_world() == "Hello, world!"
