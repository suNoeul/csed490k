#pragma once
#include <iostream>
#include <vector>
#include <mpi.h>

using namespace std;

// Define 2D board type
using Board = vector<vector<char>>;

// State constants
const char ALIVE = '#';
const char DEAD  = '.';
const int  ROOT  =  0 ;

// Define the 8 directions to check neighbors 
constexpr int dx[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
constexpr int dy[8] = {-1,  0,  1, -1, 1, -1, 0, 1};

// Environment struct
struct Env {
    int m, N, ghost, size, rank;
    int local_rows;
    int total_rows;
    int total_cols;

    int base, remainder;

    Env(int m_, int N_, int ghost_, int size_, int rank_)
        : m(m_), N(N_), ghost(ghost_), size(size_), rank(rank_) {

        base = m / size;
        remainder = m % size;

        local_rows = (rank == size - 1) ? base + remainder : base;
        total_rows = local_rows + 2 * ghost;
        total_cols = m + 2 * ghost;
    }
};

void read_input_and_distribute  (const Env& env, const Board& board, Board& local_board);
void receive_subboard           (const Env& env,       Board& local_board);
void exchange_ghost_rows        (const Env& env,       Board& local_board);
void compute_next_generation    (const Env& env, const Board& current, Board& next);
void gather_result_and_print    (const Env& env, const Board& local_board);