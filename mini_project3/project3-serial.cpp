#include <iostream>
#include <vector>

using namespace std;

// Define 2D board type
using Board = vector<vector<char>>;

// State constants
const char ALIVE = '#';
const char DEAD  = '.';

// Define the 8 directions to check neighbors 
constexpr int dx[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
constexpr int dy[8] = {-1,  0,  1, -1, 1, -1, 0, 1};


// Count alive neighbors for a given cell
int count_alive_neighbors(const Board& board, int i, int j, int m) {
    int count = 0;
    for (int k = 0; k < 8; ++k) {
        int ni = i + dx[k];
        int nj = j + dy[k];
        if (ni >= 0 && ni < m && nj >= 0 && nj < m && board[ni][nj] == ALIVE)
            ++count;
    }
    return count;
}

// Compute next generation based on current board
void compute_next_generation(const Board& current, Board& next, int m) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            int alive = count_alive_neighbors(current, i, j, m);
            if (current[i][j] == ALIVE)
                next[i][j] = (alive == 2 || alive == 3) ? ALIVE : DEAD;
            else
                next[i][j] = (alive == 3) ? ALIVE : DEAD;
        }
    }
}

// Print board
void print_board(const Board& board, int m) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            cout << board[i][j];
        }
        cout << '\n';
    }
}

int main() {
    int m, N, ghost;
    cin >> m >> N >> ghost; // ghost is ignored in serial version

    Board current(m, vector<char>(m, DEAD));
    Board next(m, vector<char>(m, DEAD));

    // Input initial board state
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            cin >> current[i][j];

    for (int gen = 0; gen < N; ++gen) {
        compute_next_generation(current, next, m);
        current.swap(next);
    }

    print_board(current, m);
    return 0;
}
