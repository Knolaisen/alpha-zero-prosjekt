import state_handler.chess_handler as sh


state = sh.ChessStateHandler()
print(state.get_all_moves())
print(state.get_board_state())
