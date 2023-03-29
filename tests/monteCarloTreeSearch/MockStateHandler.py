from abc import ABC, abstractmethod

import pytest

from monteCarloTreeSearch.state import state_handler

@pytest.fixture
def mock_state_handler():
    class MockStateHandler(state_handler):
        def is_finished(self):
            return False
        
        def get_winner(self):
            return None
        
        def get_legal_actions(self):
            return [0, 1, 2, 3, 4, 5, 6]
        
        def move(self, action):
            pass
        
        def get_state(self):
            return None

    return MockStateHandler()
