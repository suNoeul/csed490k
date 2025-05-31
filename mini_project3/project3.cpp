#include "project3.hpp"

// 마스터 프로세스: 보드 입력 받고 각 프로세스에 조각 전송 (one-to-one)
void read_input_and_distribute(const Env& env, const Board& board, Board& local_board) {   
    for (int process_rank = 0; process_rank < env.size; ++process_rank) {    
        int start = process_rank * env.base;
        int rows_to_send = (process_rank == env.size - 1) ? env.base + env.remainder : env.base;

        for (int i = 0; i < rows_to_send; ++i) {
            int row_idx = start + i;
            if (process_rank == ROOT) {
                for (int j = 0; j < env.m; ++j)
                    local_board[i + env.ghost][j + env.ghost] = board[row_idx][j];
            } else {
                // one-to-one 전송
                MPI_Send(board[row_idx].data(), env.m, MPI_CHAR, process_rank, 0, MPI_COMM_WORLD); 
            }
        }
    }
}

// 각 프로세스: 마스터로부터 보드 조각 수신 (one-to-one)
void receive_subboard(const Env& env, Board& local_board) {
    for (int i = 0; i < env.local_rows; ++i) {
        MPI_Recv(local_board[i + env.ghost].data() + env.ghost, env.m, MPI_CHAR, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void exchange_ghost_rows(const Env& env, Board& local_board) {
    int tag = 1;
    MPI_Status status;

    // g번째 ghost row 교환
    for (int g = 0; g < env.ghost; ++g) {
        // 위쪽 이웃과 교환
        if (env.rank > 0) {
            MPI_Sendrecv(
                local_board[env.ghost + g].data() + env.ghost, env.m, MPI_CHAR, env.rank - 1, tag,
                local_board[            g].data() + env.ghost, env.m, MPI_CHAR, env.rank - 1, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }
    
        // 아래쪽 이웃과 교환
        if (env.rank < env.size - 1) {
            MPI_Sendrecv(
                local_board[env.local_rows             + g].data() + env.ghost, env.m, MPI_CHAR, env.rank + 1, tag,
                local_board[env.local_rows + env.ghost + g].data() + env.ghost, env.m, MPI_CHAR, env.rank + 1, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }
    }
}

void compute_next_generation(const Env& env, const Board& current, Board& next) {
    for (int i = env.ghost; i < env.ghost + env.local_rows; ++i) {
        for (int j = env.ghost; j < env.ghost + env.m; ++j) {
            
            // Count number of alive neighbors
            int alive_neighbors = 0;
            for (int k = 0; k < 8; ++k) {
                int ni = i + dx[k];
                int nj = j + dy[k];
                if (current[ni][nj] == ALIVE) ++alive_neighbors;
            }

            // Apply Game of Life rules
            if (current[i][j] == ALIVE) 
                next[i][j] = (alive_neighbors == 2 || alive_neighbors == 3) ? ALIVE : DEAD;
            else 
                next[i][j] = (alive_neighbors == 3) ? ALIVE : DEAD;

        }
    }
}

void gather_result_and_print(const Env& env, const Board& local_board) {
    int row_size = env.m;
    vector<char> sendbuf(env.local_rows * row_size);

    for (int i = 0; i < env.local_rows; ++i)
        for (int j = 0; j < row_size; ++j)
            sendbuf[i * row_size + j] = local_board[i + env.ghost][j + env.ghost];

    if (env.rank == ROOT) {
        vector<int> recvcounts(env.size);
        vector<int> displs(env.size);
        int offset = 0;

        // recvcounts, displacements 계산
        for (int i = 0; i < env.size; ++i) {
            int rows = (i == env.size - 1) ? env.base + env.remainder : env.base;
            recvcounts[i] = rows * row_size;
            displs[i] = offset;
            offset += recvcounts[i];
        }

        vector<char> recvbuf(env.m * env.m); // 전체 보드 크기만큼

        MPI_Gatherv(sendbuf.data(), env.local_rows * row_size, MPI_CHAR,
                    recvbuf.data(), recvcounts.data(), displs.data(), MPI_CHAR,
                    ROOT, MPI_COMM_WORLD);

        // 출력
        for (int i = 0; i < env.m; ++i) {
            for (int j = 0; j < env.m; ++j)
                cout << recvbuf[i * row_size + j];
            cout << endl;
        }
    } else {
        MPI_Gatherv(sendbuf.data(), env.local_rows * row_size, MPI_CHAR,
                    nullptr, nullptr, nullptr, MPI_CHAR,
                    ROOT, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m, N, ghost;

    if (rank == 0) 
        cin >> m >> N >> ghost;
    
    // one-to-all broadcast
    MPI_Bcast(&m, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&ghost, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // 공통 환경 설정 & 각 프로세스의 로컬 보드 (ghost 포함) 생성
    Env env(m, N, ghost, size, rank);
    Board local_board(env.total_rows, vector<char>(env.total_cols));
    Board  next_board(env.total_rows, vector<char>(env.total_cols));;

    // 전체 보드는 rank 0만 가짐
    Board board;
    if (rank == ROOT) {
        board.resize(m, vector<char>(m));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                cin >> board[i][j];

        read_input_and_distribute(env, board, local_board);
    } else {
        receive_subboard(env, local_board);
    }

    // ghost 교환, 세대 계산, 결과 수집 등 구현
    for (int gen = 0; gen < N; ++gen) {
        exchange_ghost_rows(env, local_board);
        compute_next_generation(env, local_board, next_board);
        local_board.swap(next_board);
    }

    gather_result_and_print(env, local_board);

    MPI_Finalize();
    return 0;
}

