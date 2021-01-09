/*----------------------------------------------------
NOTES:

1. the matrix mat is partioned by columns so that each process gets n / p columns. Therefore,
   this program assumes n is evenly divisible by   p (number of processes).

2. If there is no edge between any two vertices the weight is the constant INFINITY.



 *-----------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS // to close warnings.
#define INFINITY 1000000

int global_distance_matrix[8][8] = {
{0,40,15,INFINITY,INFINITY,INFINITY,35,INFINITY},
{40,0,0,100,INFINITY,INFINITY,25,INFINITY},
{15,0,0,20,10,50,INFINITY,50},
{INFINITY,100,20,0,10,INFINITY,45,INFINITY},
{INFINITY,INFINITY,10,10,0,30,50,30},
{INFINITY,INFINITY,50,INFINITY,30,0,INFINITY,0},
{35,25,INFINITY,45,50,INFINITY,0,0},
{INFINITY,INFINITY,50,INFINITY,30,0,0,0}
};  // global distance matrix


// some signatures
int init_number_of_vertices(int task_id, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void init_matrix(int loc_mat[], int n, int loc_n, MPI_Datatype blk_col_mpi_t,
    int task_id, MPI_Comm comm);
void Dijkstra_Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
    int task_id, int loc_n);
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
    MPI_Comm comm);
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n);
void print_distance_matrix(int global_mat[]);
void Print_dists(int global_dist[], int n);



// driver => main

int main(int argc, char** argv) {
    int* loc_mat, * loc_dist, * loc_pred, * global_dist = NULL, * global_pred = NULL;
    int task_id, nmb_of_process, loc_n, n;
    MPI_Comm comm;  // Communicator that describe a group of process
    MPI_Datatype blk_col_mpi_t;  // to  store MPI derived data type

    MPI_Init(NULL, NULL);  // initialize MPI
    comm = MPI_COMM_WORLD; //all processes that were started together by mpiexec
    // each process assigned a unique integer rank between 0 and number of tasks - 1


    MPI_Comm_rank(comm, &task_id); // Returns the rank of the calling MPI process within the specified communicator

    MPI_Comm_size(comm, &nmb_of_process); // Returns the total number of MPI processes in the specified communicator

    n = init_number_of_vertices(task_id, comm);  // initialize number of vertices in graph. parameters = task_id : process id, comm: communicator

    if (n % nmb_of_process != 0) // check if n/nmb_of_process  is integer
        nmb_of_process = 4;

    loc_n = n / nmb_of_process;  // number cols in the block column 

    loc_mat = malloc(n * loc_n * sizeof(int));  // calling process's submatrix, Allocated submatrix size = n X number of columns (loc_n)

    loc_dist = malloc(loc_n * sizeof(int)); // to store shortest distances

    loc_pred = malloc(loc_n * sizeof(int)); //  loc_pred[v] = predecessor of v on a shortest path from source to v

    blk_col_mpi_t = Build_blk_col_type(n, loc_n);  // Build an MPI_Datatype that represents a block column of a matrix


    // if master thread
    if (task_id == 0) {
        global_dist = malloc(n * sizeof(int));  // processes save found dist into this.
        global_pred = malloc(n * sizeof(int));  // global_pred[v] = predecessor of v on a shortest path from source to v (global)
    }
    init_matrix(loc_mat, n, loc_n, blk_col_mpi_t, task_id, comm);  // init distance matrix

    if (task_id == 0) // if master thread
    {

        printf("\n\n");
        printf(" ______________________________ Disjkstra Algorithm MPI Implementation ______________________________");
        printf("\n\n");
        printf(" ______________________________ MELIH GORGULU - 161180032 ______________________________");
        printf("\n");
        printf("\n");

        printf("__________ %d processes will be used for Dijkstra Algorithm__________", nmb_of_process);

        printf("\n");
        printf("\n");

        printf("Input Distance Matrix:\n");
        print_distance_matrix(loc_mat);
        printf("\n\n");
    }

    Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, n, comm);  // run dijsktra algorithm

    // Gather the results from Dijkstra 
    // Gathers distinct messages from each task in the group to a single destination task
    // messages will be gathered into task0 (root task)


    MPI_Gather(loc_dist, loc_n, MPI_INT, global_dist, loc_n, MPI_INT, 0, comm);
    MPI_Gather(loc_pred, loc_n, MPI_INT, global_pred, loc_n, MPI_INT, 0, comm);

    // printing results
    if (task_id == 0) {
        Print_dists(global_dist, n); // print the distance matrix
        free(global_dist);
        free(global_pred);
    }

    // free allocated memory space
    free(loc_mat);
    free(loc_pred);
    free(loc_dist);
    MPI_Type_free(&blk_col_mpi_t); // Deallocate the blk_col_mpi_t
    MPI_Finalize();  // Terminate the MPI execution environment.


    return 0;
}






int init_number_of_vertices(int task_id, MPI_Comm comm) { // task_id : calling process rank, comm Communicator containing all calling processes

    int n = 8; // number of vertices in the graph = 8
    // broadcast the number of vertices to the all proccess.
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);  // n: number of vertices, count:1,MPI_INT: MPI type handle (int type message).
    return n;
}




// Purpose:   Build an MPI_Datatype that represents a block column of a matrix

MPI_Datatype Build_blk_col_type(int n, int loc_n) {
    MPI_Aint lb, extent;  // type of a variable able to contain a memory address
    MPI_Datatype block_mpi_t;
    MPI_Datatype first_bc_mpi_t;
    MPI_Datatype blk_col_mpi_t;

    MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t); //contiguous datatype
    MPI_Type_get_extent(block_mpi_t, &lb, &extent); // Get the lower bound and extent for a Datatype, lb: lower bound, extent: Extent

    // MPI_Type_vector(numblocks, blocklength, stride, oldtype, *newtype) => blocklength : number of elements in each block,stride:number of elements between start of each block
    MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t); // creating vector data type

    // This call is needed to get the right extent of the new datatype 
    MPI_Type_create_resized(first_bc_mpi_t, lb, extent, &blk_col_mpi_t);

    MPI_Type_commit(&blk_col_mpi_t); // committing data type

    MPI_Type_free(&block_mpi_t);
    MPI_Type_free(&first_bc_mpi_t);

    return blk_col_mpi_t;
}


//  Read nxn matrix on process 0, and distribute it among the processes so that each process gets a block column with n rows and n/p columns


// In args:   n:  the number of rows/cols in the matrix and the submatrices
//            loc_n = n/p:  the number of columns in the submatrices
//            blk_col_mpi_t:  the MPI_Datatype used on process 0
//            task_id:  the caller's rank in comm
//            comm:  Communicator 

// Out arg:   loc_mat:  the calling process' submatrix 

void init_matrix(int loc_mat[], int n, int loc_n,

    // create the MPI_Datatype
    MPI_Datatype blk_col_mpi_t, int task_id, MPI_Comm comm) {
    int* mat = NULL, i, j;

    int Inf = 1000000; // represent infinity

    if (task_id == 0) {
        mat = malloc(n * n * sizeof(int));
        // Let's create the matrix to be processed by tasks
        // use predefined global distance matrix for it

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {

                mat[i * n + j] = global_distance_matrix[i][j];

            }


    }

    // SCATTER the distance matrix to the procees.
    // recive buffer = loc_mat
    // each buffers size = n * loc_n
    MPI_Scatter(mat, 1, blk_col_mpi_t, loc_mat, n * loc_n, MPI_INT, 0, comm);  // source = 0 (root process)

    if (task_id == 0) free(mat);
}



/*
 -->            loc_dist: loc_dist[v] = shortest distance from the source to each vertex v
 -->            loc_pred: loc_pred[v] = predecessor of v on a shortest path from source to v
 -->            loc_known: loc_known[v] = 1 if vertex has been visited, 0 else

 */

 // Dijkstra_Init : Initialize all the matrices

void Dijkstra_Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
    int task_id, int loc_n) {
    int loc_v;

    if (task_id == 0) // if first (starting) vertice.
        loc_known[0] = 1;
    else
        loc_known[0] = 0;

    for (loc_v = 1; loc_v < loc_n; loc_v++)
        loc_known[loc_v] = 0;

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        loc_dist[loc_v] = loc_mat[0 * loc_n + loc_v];
        loc_pred[loc_v] = 0;
    }
}



// __________________DIJKSTRA FUNCTION__________________

//  Purpose: compute all the shortest paths from the source vertex 0 to all vertices v

// Inputs : 

// --> loc_mat:  local matrix containing edge costs between vertices
// --> loc_n : local number of vertices 
// --> n : total number of vertices
// --> comm : the communicator

// Outputs:
// --> Out args : loc_dist: loc_dist[v] = shortest distance from the source to each vertex v
// --> loc_pred : loc_pred[v] = predecessor of v on a shortest path from source to v

void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
    MPI_Comm comm) {

    int i, loc_v, loc_u, glbl_u, new_dist, task_id, dist_glbl_u;
    int* loc_known;
    int my_min[2];
    int glbl_min[2];

    MPI_Comm_rank(comm, &task_id); // get process rank
    loc_known = malloc(loc_n * sizeof(int));

    Dijkstra_Init(loc_mat, loc_pred, loc_dist, loc_known, task_id, loc_n); // init task matrice

    /* Run loop n - 1 times since we already know the shortest path to global
       vertex 0 from global vertex 0 */
    for (i = 0; i < n - 1; i++) {
        loc_u = Find_min_dist(loc_dist, loc_known, loc_n);  // finding mind dist


        // check if INFINITY or NOT
        if (loc_u != -1) {
            my_min[0] = loc_dist[loc_u]; // my_min[0] = distance value, my_min[1] = vertex
            my_min[1] = loc_u + task_id * loc_n;
        }
        else {
            my_min[0] = INFINITY;
            my_min[1] = -1;
        }

        /* Get the minimum distance found by the processes and store that
           distance and the global vertex in glbl_min
        */

        // NOTE: MPI_MINLOC used to global minumum
        MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);  // sendbuf:my_min,recvbuf:glbl_min,count:1,MPI_INT

        dist_glbl_u = glbl_min[0];
        glbl_u = glbl_min[1];

        // This test is to assure that loc_known is not accessed with -1 
        if (glbl_u == -1)
            break;

        /* Check if global u belongs to process, and if so update loc_known */
        if ((glbl_u / loc_n) == task_id) {
            loc_u = glbl_u % loc_n;
            loc_known[loc_u] = 1;
        }

        /* For each local vertex (global vertex = loc_v + task_id * loc_n)
           Update the distances from source vertex (0) to loc_v. If vertex
           is unmarked check if the distance from source to the global u + the
           distance from global u to local v is smaller than the distance
           from the source to local v
         */
        for (loc_v = 0; loc_v < loc_n; loc_v++) {
            if (!loc_known[loc_v]) {
                new_dist = dist_glbl_u + loc_mat[glbl_u * loc_n + loc_v];
                if (new_dist < loc_dist[loc_v]) {
                    loc_dist[loc_v] = new_dist;
                    loc_pred[loc_v] = glbl_u;
                }
            }
        }
    }
    free(loc_known);
}




// __________________ FIND_MIN_DIST FUNCTION __________________

// Purpose:  finds the minimum distance between the vertice assigned to the process that calls the method and the source vertice.

// Inputs:
// -- > loc_dist:  array with distances from source 0
// -- > loc_known : array with values 1 if the vertex has been visited
// -- > 0 if not
// -- > loc_n : local number of vertices

// Output:
// -- > loc_u: the vertex with the smallest value in loc_dist,-1 if all vertices are already known


int Find_min_dist(int loc_dist[], int loc_known[], int loc_n) {
    int loc_u = -1, loc_v;
    int shortest_dist = INFINITY; // assume infinity in the first step

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        // if not visited
        if (!loc_known[loc_v]) {
            if (loc_dist[loc_v] < shortest_dist) {

                // update shortest distance
                shortest_dist = loc_dist[loc_v];
                loc_u = loc_v;
            }
        }
    }
    return loc_u;
}



// Print the contents of the matrix

void print_distance_matrix(int mat[]) {
    int i, j;
    printf("\n");
    printf("   ");

    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++)
            if (global_distance_matrix[i][j] == INFINITY)
                printf(" Inf ");
            else
                printf(" %d ", global_distance_matrix[i][j]);


        printf("\n");
        printf("   ");
    }

    printf("\n");
}


// Print the length of the shortest path from 0 to each
// uses global dist 

void Print_dists(int global_dist[], int n) {
    int v = 0;

    printf("Minimum distances from node 1:\n\n");


    for (v = 0; v < n; v++) {

        if (v == 0) {
            printf("%3d    %4d\n", v + 1, global_dist[v]);
            continue;
        }
        if (global_dist[v] == INFINITY) {
            printf("%3d    %5s\n", v + 1, "inf");
        }
        else
            printf("%3d    %4d\n", v + 1, global_dist[v]);
    }
    printf("\n");
}