def print_board(board):
    """
    Prints the Sudoku board in a grid format.
    0 indicates an empty cell.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 represents an empty cell.

    Returns:
    None
    """
    for row_idx, row in enumerate(board):
        # Print a horizontal separator every 3 rows (for sub-grids)
        if row_idx % 3 == 0 and row_idx != 0:
            print("- - - - - - - - - - -")

        row_str = ""
        for col_idx, value in enumerate(row):
            # Print a vertical separator every 3 columns (for sub-grids)
            if col_idx % 3 == 0 and col_idx != 0:
                row_str += "| "

            if value == 0:
                row_str += ". "
            else:
                row_str += str(value) + " "
        print(row_str.strip())


def find_empty_cell(board):
    """
    Finds an empty cell (indicated by 0) in the Sudoku board.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 represents an empty cell.

    Returns:
    tuple or None:
        - If there is an empty cell, returns (row_index, col_index).
        - If there are no empty cells, returns None.
    """
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 0:
                return (row, col)
    return None


def is_valid(board, row, col, num):
    """
    Checks if placing 'num' at board[row][col] is valid under Sudoku rules:
      1) 'num' is not already in the same row
      2) 'num' is not already in the same column
      3) 'num' is not already in the 3x3 sub-box containing that cell

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board.
    row (int): Row index of the cell.
    col (int): Column index of the cell.
    num (int): The candidate number to place.

    Returns:
    bool: True if valid, False otherwise.
    """
    row_nums = [0] * len(board)
    row_nums[num - 1] = 1
    col_nums = [0] * len(board)
    col_nums[num - 1] = 1
    for i in range(len(board)):
        if board[row][i] != 0:
            if row_nums[board[row][i] - 1] != 0:
                return False
            row_nums[board[row][i] - 1] = 1
        if board[i][col] != 0:
            if col_nums[board[i][col] - 1] != 0:
                return False
            col_nums[board[i][col] - 1] = 1

    row_start = row - (row % 3)
    col_start = col - (col % 3)
    
    square_nums = [0] * len(board)
    square_nums[num - 1] = 1
    for i in range(row_start, row_start + 3):
        for j in range(col_start, col_start + 3):
            if board[i][j] != 0:
                if square_nums[board[i][j] - 1] != 0:
                    return False
                square_nums[board[i][j] - 1] = 1
    
    return True


def solve_sudoku(board):
    """
    Solves the Sudoku puzzle in 'board' using backtracking.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 indicates an empty cell.

    Returns:
    bool:
        - True if the puzzle is solved successfully.
        - False if the puzzle is unsolvable.
    """
    cell = find_empty_cell(board)

    if cell == None:
        return is_solved_correctly(board)
    
    for i in range(1, 10):
        if not is_valid(board, cell[0], cell[1], i):
            continue

        board[cell[0]][cell[1]] = i
        
        if solve_sudoku(board):
            return True
    
    board[cell[0]][cell[1]] = 0

    return False

def check_range(board, row_start, row_end, col_start, col_end):
    nums = [0] * len(board)
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            if board[row][col] != 0:
                if nums[board[row][col] - 1] != 0:
                    return False
                nums[board[row][col] - 1] = 1
    
    for num in nums:
        if num != 1:
            return False
        
    return True

def is_solved_correctly(board):
    """
    Checks that the board is fully and correctly solved:
    - Each row contains digits 1-9 exactly once
    - Each column contains digits 1-9 exactly once
    - Each 3x3 sub-box contains digits 1-9 exactly once

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board.

    Returns:
    bool: True if the board is correctly solved, False otherwise.
    """
    for i in range(len(board)):
        if not check_range(board, 0, len(board), i, i + 1) or not check_range(board, i, i + 1, 0, len(board)):
            return False
        
    for i in range(0, len(board), 3):
        for j in range(0, len(board), 3):
            if not check_range(board, i, i + 3, j, j + 3):
                return False
            
    return True


if __name__ == "__main__":
    # Example usage / debugging:
    example_board = [
        [7, 8, 0, 4, 0, 0, 1, 2, 0],
        [6, 0, 0, 0, 7, 5, 0, 0, 9],
        [0, 0, 0, 6, 0, 1, 0, 7, 8],
        [0, 0, 7, 0, 4, 0, 2, 6, 0],
        [0, 0, 1, 0, 5, 0, 9, 3, 0],
        [9, 0, 4, 0, 6, 0, 0, 0, 5],
        [0, 7, 0, 3, 0, 0, 0, 1, 2],
        [1, 2, 0, 0, 0, 7, 4, 0, 0],
        [0, 4, 9, 2, 0, 6, 0, 0, 7],
    ]

    print("Debug: Original board:\n")
    print_board(example_board)
    print(solve_sudoku(example_board))
