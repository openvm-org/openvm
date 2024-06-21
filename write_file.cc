#include <iostream>
#include <string>
#include <fstream>
#define MAX 1000000

using namespace std;

int main(int argc, char *argv[])
{

    int i = stoi(argv[1]);
    ofstream fout;
    fout.open("tmp.afi");
    fout << "TABLE_ID big_tree\n";
    fout << "INDEX_BYTES 32\n";
    fout << "DATA_BYTES 1024\n";
    for (int j = 0; j < MAX; j++)
    {
        string s = to_string(MAX * i + j);
        fout << "INSERT " << s << " " << s << "\n";
    }
}