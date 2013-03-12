

int * load_int(char *file, int * size) {

  FILE *fp = fopen(file, "r");
  if (fp == NULL) { // check to see if file opens; if NULL, failed
    printf("Error opening file. \n");
    return 0;
  }

  fscanf(fp, "%d", size); // First line of input file is designated as the total number of elements in file; store it into "size"

  int * Array = malloc((*size)*(*size) * sizeof(int));
  if (Array == NULL) {
    printf("Out of memory!\n");
    return 0;
  }
  int i=0;
  while (! feof(fp)) {
    fscanf(fp, "%d", &Array[i]);// Store all numbers into Array
    i++;
  }

  fclose(fp);
  return(Array); // Return array for main.c
}

float * load_float(char *file, int * size) {

  FILE *fp = fopen(file, "r");
  if (fp == NULL) { // check to see if file opens; if NULL, failed
    printf("Error opening file. \n");
    return 0;
  }

  fscanf(fp, "%d", size); // First line of input file is designated as the total number of elements in file; store it into "size"

  float * Array = malloc((*size)*(*size) * sizeof(float));
  if (Array == NULL) {
    printf("Out of memory!\n");
    return 0;
  }
  int i=0;
  while (! feof(fp)) {
    fscanf(fp, "%f", &Array[i]);// Store all numbers into Array
    i++;
  }

  fclose(fp);
  return(Array); // Return array for main.c
}
