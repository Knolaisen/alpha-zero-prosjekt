import pygame
import chess
import sys
import chess.svg



def visualize(board):
    # Set up the pygame window
    WINDOW_SIZE = (800, 800)
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Chessboard")

    # Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = 	(255, 0, 0)
    GREEN = (0, 255, 0)
    # Main game loop
    selected_square = None
    move_square = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONUP:
                # Get the position of the mouse click
                x, y = event.pos
                # Convert the pixel coordinates to chess square coordinates
                file = int(x / (WINDOW_SIZE[0] / 8))
                rank = int((WINDOW_SIZE[1] - y) / (WINDOW_SIZE[1] / 8))
                

                #implement such that second 
                if (selected_square!=None ):
                    move_square = chess.square(file,rank)
                    # print(f"move from square {chess.square_name(square)} to square {chess.square_name(move_square)} ")
                    movesquare_string = str(chess.square_name(selected_square)+ chess.square_name(move_square))
                    if(chess.Move.from_uci((movesquare_string)) in board.legal_moves):
                        board.push_uci(movesquare_string)
                        print(f"Registered move {movesquare_string}")
                    else:
                        print(f"{movesquare_string} Not a legal move")

                    
                else: 
                    square = chess.square(file, rank)
                    # Set the selected square to the clicked square
                    selected_square = square
                    move_square = None
                    # Print the square that was clicked
                    print(f"You clicked on square {chess.square_name(square)}")
        # Draw the board on the screen
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                # Get the piece on the square
                piece = board.piece_at(square)
                # Determine the color of the square
                if (file + rank) % 2 == 1:
                    square_color = WHITE
                else:
                    square_color = BLACK
                # If the square is selected, highlight it in red
                if selected_square == square:
                    square_color = RED
                    
                elif move_square == square:
                    square_color = GREEN
                    selected_square = None
                # Draw the square on the screen
                pygame.draw.rect(screen, square_color, pygame.Rect(file * 100, (7 - rank) * 100, 100, 100))
                
                # Draw the piece on the square, if there is one
                if piece is not None:
                    if piece.color == chess.WHITE:
                        if piece.piece_type == chess.PAWN:
                            piece_img = pygame.image.load("src\Imgs\w_pawn.png")
                        elif piece.piece_type == chess.ROOK:
                            piece_img = pygame.image.load("src\Imgs\w_rook.png")
                        elif piece.piece_type == chess.KNIGHT:
                            piece_img = pygame.image.load("src\Imgs\w_knight.png")
                        elif piece.piece_type == chess.BISHOP:
                            piece_img = pygame.image.load("src\Imgs\w_bishop.png")
                        elif piece.piece_type == chess.QUEEN:
                            piece_img = pygame.image.load("src\Imgs\w_queen.png")
                        else:
                            piece_img = pygame.image.load("src\Imgs\w_king.png")
                    else:
                        if piece.piece_type == chess.PAWN:
                            piece_img = pygame.image.load("src\Imgs\\b_pawn.png")
                        elif piece.piece_type == chess.ROOK:
                            piece_img = pygame.image.load("src\Imgs\\b_rook.png")
                        elif piece.piece_type == chess.KNIGHT:
                            piece_img = pygame.image.load("src\Imgs\\b_knight.png")
                        elif piece.piece_type == chess.BISHOP:
                            piece_img = pygame.image.load("src\Imgs\\b_bishop.png")
                        elif piece.piece_type == chess.QUEEN:
                            piece_img = pygame.image.load("src\Imgs\\b_queen.png")
                        else:
                            piece_img = pygame.image.load("src\Imgs\\b_king.png")
                    screen.blit(piece_img, pygame.Rect(file * 100 + 40, (7 - rank) * 100 +10, 100, 100))
        pygame.display.flip()
        # Pause for a short period to prevent the window from closing immediately
        pygame.time.wait(10)
    # Close the Pygame window
    pygame.quit()
    sys.exit()

board = chess.Board()


visualize(board)