"""pytest config — add repo root to sys.path so `import config` etc. works."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
