#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<ctype.h>
void getinput(char []);
void create_question();
int removeSpaces(char text[600]);
void lower(char strr[]);

int main()
{
  	char input[600];
  	printf("This a chatbot in C language\n");
  	printf("Version 2.0\n");
  	printf("My name is Alex\n");
  	do
  	{
    		printf(">>");
    		gets(input);
			lower(input);
    		getinput(input);
   	} while(1);
}
void create_question()
{
  char array[600];
  char input[600];
  char ans1[600];
  char ans2[600];
  char ans3[600];
  
  FILE  *fp=fopen("c.txt","a");

   fgets(array,600,fp);
     printf("sorry its not in the directory\n");
     printf("do you want to add it\n");
     printf("yes or no\n");
     if(strcmp(gets(input),"yes")==0)
     {
       printf("add the question\n");
       gets(input);
	   lower(input);
       printf("add the answers\n");
       printf("add the answer1\n");
       gets(ans1);
       printf("add the answer2\n");
       gets(ans2);
       printf("add the answer3\n");
       gets(ans3);
	   strcat(input,"|");
	   strcat(input,ans1);
	   strcat(input,"|");
	   strcat(input,ans2);
	   strcat(input,"|");
	   strcat(input,ans3);
	   strcat(input,"\n");
       fprintf(fp,"%s",input);
      }
    fclose(fp);
    }

void getinput(char input[600])
{
    int exists=0;
	srand(time(0));
	char show[500];
	char array[600];
   	FILE *fp=fopen("c.txt" ,"r");
   	while(!feof(fp))
   	{
     		fgets(array,600,fp);
                
                        
     		if(strncmp(array,input,strlen(input))==0)
     		 {	
			strtok(array, "|");
			switch (rand() % 3){
				case 0:
				{
					strcpy(show,strtok(NULL, "|"));
					puts(show);
					removeSpaces(show);
				    break;
				}
				case 1:
				{
					strtok(NULL, "|");
					strcpy(show,strtok(NULL, "|"));
					puts(show);
					removeSpaces(show);					
					break;
				}
				case 2:
				{
					strtok(NULL, "|");
					strtok(NULL, "|");
					strcpy(show,strtok(NULL, "|"));
					puts(show);
					removeSpaces(show);
					break;
				}
			}exists=1;
			break;//remove
     		}
   	}
  	fclose(fp);
        
 if(!exists)
 {
    create_question();     
  }  
}

int removeSpaces(char text[600])
{ 
	 char  blank[1000];
   int c = 0, d = 0;
 
   while (text[c] != '\0') {
      if (text[c] == ' ') {
		  text[c] = '_';
         int temp = c + 1;
         if (text[temp] != '\0') {
            while (text[temp] != ' ' && text[temp] != '\0') {
               if (text[temp] == ' ') {
				   text[temp] = '_';
                  c++;
               }  
               temp++;
            }
         }
      }
      blank[d] = text[c];
      c++;
      d++;
   }
 
   blank[d] = '\0';
	    char command[] = "espeak ";		
		strcat(command,blank);
		system(command);
}
  
 void lower(char strr[])
 {
	int i;
	for(i=0;strr[i];i++)
	{
		strr[i]=tolower(strr[i]);
	}
 }
