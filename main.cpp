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
	Linear(int in_features, int out_features){
		in_dim = in_features;
		out_dim = out_features;

		random_device rd;
		mt19937 gen(rd());
		double square_root_k = sqrt(in_features);
		uniform_real_distribution<> dist(-square_root_k, square_root_k);
		for(int i = 0; i < out_features; i++){
			vector<double> temp = {};
			for(int o = 0; o < in_features; o++){
				temp.push_back(dist(gen));
			}
			w.push_back(temp);
		}
		for(int i = 0; i < out_features; i++){
			b.push_back(dist(gen));
		}
	}

	void print_weight(){
		cout << "w = {" << "\n";
		for(vector<double> i : w){
			for(double o : i){
				cout << o << ',';
			}
			cout << "\n";
		}
		cout << "}" << endl;
	}
	void print_bias(){
		cout << "b = {" ;
		for(double i : b){
			cout << i << ',';
		}
		cout << "}" << endl;
	}
	
	vector<double> forward(vector<double> x){
		vector<double> OUT;
		for(int i = 0; i < out_dim; i++){
			vector<double> temp_node;
			double temp;
			for(int o = 0; o < in_dim; o++){
				temp += w[i][o] * x[0] + b[o];
			}
			OUT.push_back(temp);
		}
		return OUT;
	}
	
	~Linear(){
	}
};

class DQN{
public:
	Linear layer1 = Linear(3,3);
	DQN(){
	}
	~DQN(){
	}
};

class TTT{
private:
	vector<vector<double>> game_screen;
	int width;
	int height;
public:
	TTT(){
		u = 3;
		v = 3;
		//le
	}

}

double print_vector(vector<double> arg_vec){
	for(double unit : arg_vec){
		cout << unit << ",";
	}
	cout << endl;
}

int main(){
	DQN net;
	net.layer1.print_weight();
	return 0;
}
