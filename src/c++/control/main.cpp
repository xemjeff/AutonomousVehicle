#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "control.h"

void printHeader() 
{
   printf("Car control (Burlington Robotics)\n");
   printf("---------------------------------\n");
}

void printHelp()
{
   printf("Commands:\n");
   printf("s : speed\n");
   printf("d : direction\n");
   printf("n : neutral\n");
   printf("l : left\n");
   printf("r : right\n");
   printf("q : quit\n");
   printf("? : help\n");
}

int main(int argc, char *argv[])
{
   bool done = false;
   CarControl* car = new CarControl();
   if (!car->isReady())
   {
      printf("gpio initialization failed. run as sudo?\n");
      return -1;
   }

   printHeader();
   printHelp();
   double speed     = 0.0;
   double direction = 0.0;

   char instr[128];
   char delim = ' ';

   while(!done) {
    //read in the command
     printf("> ");
     fgets(instr, 127, stdin);

    //duplicate and strip the newline
    char* str = strdup(instr);
    int length = strlen(str);
    if (str[length - 1] == '\n') 
      str[length - 1] = '\0';     

     //parse into words, separated by space
    char* cmd = strsep(&str, &delim);
    char* arg = strsep(&str, &delim);

    //process command
    if ( strcmp(cmd,"s") == 0) {
      speed = atof(arg);
      car->setSpeed(speed);
    }
    else if ( strcmp(cmd,"d") == 0) {
      direction = atof(arg);
      car->setDirection(direction);
    }
    else if ( strcmp(cmd,"n") == 0) {
      car->neutral();
    }
    else if ( strcmp(cmd,"l") == 0) {
      car->left();
    }
    else if ( strcmp(cmd,"r") == 0) {
      car->right();
    }
    else if ( strcmp(cmd,"+") == 0) {
      car->faster();
    }
    else if ( strcmp(cmd,"-") == 0) {
      car->slower();
    }
    else if ( strcmp(cmd,"q") == 0) {
      printf("quit");
      done = true;
    }
    else if ( strcmp(cmd,"?") == 0 || strcmp(cmd,"h") == 0)
      printHelp();
    else
      printf("unknown command");
    printf("\n");
   }
   delete car;
   return 0;
}
