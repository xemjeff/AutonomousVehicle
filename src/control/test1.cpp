#include <stdio.h>
#include <unistd.h>
#include "control.h"

void printHeader() 
{
   printf("Test1 (Burlington Robotics)\n");
   printf("---------------------------\n");
   printf("This runs a standard pattern\n");
   printf("Fwd(0.5s), Right(0.5s), Fwd(0.5s), Left(0.5s), Wait(1s), Rev(0.5s)\n");
   printf("\nThe car should stop when the program exits\n");
}

void pause(double t)
{
  unsigned ut = t*1E6;
  usleep(ut);
}

int main(int argc, char *argv[])
{
  CarControl* car = new CarControl();
  if (!car->isReady())
  {
    printf("gpio initialization failed. run as sudo?\n");
    return -1;
  }

  printHeader();

  double stepDelay = 0.5;
  double revDelay = 0.05;

  car->neutral();
  printf("Car Starting in 2 sec..\n");
  pause(2.0);
  printf("Go!\n");

  printf("Fwd\n");
  car->setSpeed(0.25);
  pause(stepDelay);

  printf("Right\n");
  car->right();
  pause(stepDelay);

  printf("Fwd\n");
  car->setSpeed(0.25);
  pause(stepDelay); 

  printf("Left\n");
  car->left();
  pause(stepDelay);

  printf("Fwd\n");
  car->setSpeed(0.25);
  pause(stepDelay);

  printf("Neutral\n");
  car->neutral();
  pause(2*stepDelay);

  //reverse requires the sequence of -ve speed, then neutral, then reverse speed (also negative)

  printf("Reverse\n");
  car->setSpeed(-0.1);
  pause(revDelay);
  car->neutral();
  pause(revDelay);

  car->setSpeed(-0.25);
  pause(stepDelay);

  car->neutral();
  pause(stepDelay);

  printf("Done.");

  delete car;
  return 0;
}
