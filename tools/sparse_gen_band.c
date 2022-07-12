/**
 * @brief  random based sparse matrix generation tool
 * @author Zhuoqiang Guo <gzq9425@qq.com> 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

inline static int random_int(int m)
{
  return rand() % m;
}

inline float random_float()
{
  return (float)rand() / RAND_MAX;
}

typedef struct
{
  int x;
  int y;
  float v;
} point_t;

static int row_first_cmp(const point_t *a, const point_t *b)
{
  if (a->x != b->x)
    return a->x - b->x;
  return a->y - b->y;
}

int main(int argc, char *argv[])
{
  time_t t;
  srand((unsigned)time(&t));
  int64_t m, n;
  double rate;
  int64_t ub, lb;
  if (argc == 4)
  {
    m = atol(argv[1]);
    ub = atol(argv[2]);
    lb = atol(argv[3]);
  }
  else if (argc == 3)
  {
    m = atoi(argv[1]);
    ub = atol(argv[2]);
    lb = ub;
  }
  else if (argc == 2)
  {
    m = atoi(argv[1]);
    lb = ub = 0;
  }
  else
  {
    printf("Usage : sparse_gen_diag m [ub] [lb] \n\tub:upper band num, lb:lower band num\n\tif not specified, diagnoal matrix will be generate\n");
    exit(0);
  }
  if (lb > m - 1 || ub > m - 2)
  {
    printf("wrong argument \n");
    exit(-1);
  }
  size_t number = m + (2 * m - ub - 1) / 2 * ub + (2 * m - lb - 1) / 2 * lb;
  point_t *points = malloc(sizeof(point_t) * number);
  for (int i = 0, j = 0; i < m; i++)
  {
    points[j].x = i + 1;
    points[j].y = i + 1;
    points[j++].v = random_float() + .1f;

    //upper band
    for (int u = i + 1; u <= ub + i; u++)
    {
      if (u < m)
      {
        points[j].x = i + 1;
        points[j].y = u + 1;
        points[j++].v = random_float() + .1f;
      }
    }
    //lower band
    for (int l = i - 1; l >= i - lb; l--)
    {
      if (l >= 0)
      {
        points[j].x = i + 1;
        points[j].y = l + 1;
        points[j++].v = random_float() + .1f;
      }
    }
  }
  n = m;
  //sort
  int64_t count = number;
  //output
  char filename[100];
  sprintf(filename, "Matrix/band_%ld_%ld_%ld.mtx", m, ub, lb);
  FILE *fp = fopen(filename, "w");
  if (fp == NULL)
  {
    printf("output file open error!!!");
    exit(-1);
  }
  fprintf(fp, "%ld %ld %ld\n", m, n, count);
  for (int i = 0; i < count; i++)
  {
    fprintf(fp, "%d %d %.6f\n", points[i].x, points[i].y, points[i].v);
  }
  printf("%ld %ld %ld \n", m, n, count);

  free(points);
}
