#include<iostream>
#include<vector>
#include<random>

using namespace std;

class Linear{
private:
	vector<vector<double>> w;
	vector<double> b;
	int in_dim;
	int out_dim;
public:
	Linear(int in_dim, int out_dim){
		random_device rd;
		mt19937 gen(rd());
		square_root_k = sqrt(in_dim);
		uniform_real_distribution<> dist(-square_root_k, square_root_k);
		for(int i = 0; i < out_dim; i++){
			for(int o = 0; o < in_dim; o++){
				
			}
		}
		
	}
}

int main(){

{
