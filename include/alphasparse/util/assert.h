#pragma once

/**
 * @brief header for assert utils
 */ 

#define assert_exit(express,message)\
    if(express){\
        printf("%s\n",message);\
        exit(-1);\
    }
    