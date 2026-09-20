import unittest
from AlphaGo import go
from multiprocessing import Process
from interface.gtp_wrapper import run_gtp


class PassPlayer(object):
    def get_move(self, state):
        return go.PASS


def _stdin_simulator():
    # Module level, not nested in the test method: multiprocessing.Process on Windows
    # uses the "spawn" start method (unlike Unix's "fork"), which pickles the Process
    # and its args to hand off to the child - a nested/local function isn't picklable
    # (pickle locates a function by its qualified module path, which a closure doesn't
    # have), so this has to be a real module-level function to work cross-platform.
    return "\n".join([
        "1 name",
        "2 boardsize 19",
        "3 clear_board",
        "4 genmove black",
        "5 genmove white",
        "99 quit"])


class TestGTPProcess(unittest.TestCase):

    def test_run_commands(self):
        gtp_proc = Process(target=run_gtp, args=(PassPlayer(), _stdin_simulator))
        gtp_proc.start()
        gtp_proc.join(timeout=1)


if __name__ == '__main__':
    unittest.main()
