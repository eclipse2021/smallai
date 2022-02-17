#include<iostream>

#include<vector>
#include<queue>

#include<random>

using namespace std;

//gradient tape

typedef struct{
	double value;
	double *self;
	double *y;
}grad;

class Gradient_tape{	//?
public:
	vector<grad>;
	Gradient_tape(){
	}
}


//activations

vector<double> relu(vector<double> x){
	vector<double> OUT;
	for(double node : x){
		if(node <= 0) OUT.push_back(0);
		else OUT.push_back(node);
	}
	return OUT;
}

double relu(double x){
	if(x <= 0) return 0;
	else return x;
}

vector<double> sigmoid(vector<double> x){
	vector<double> OUT;
	for(double node : x){
		OUT.push_back(1/(1+exp(-1.0 * node)));
	}
	return OUT;
}

double sigmoid(double x){
	return 1/(1+exp(-1.0 * x));
}

//Replaymemory

typedef struct{
	vector<vector<double>> state;
	vector<double> action;
	double reward;
}ReplayBatch;

class ReplayMemory{
private:
	queue<ReplayBatch> memory;
	int capacity;
public:
	ReplayMemory(int arg_capacity){
		this->capacity = arg_capacity;
	}
	void push(vector<vector<double>> arg_state, vector<double> arg_action, double arg_reward){
		if(this->memory.size() < capacity){
			ReplayBatch temp = {arg_state, arg_action, arg_reward};
			this->memory.push(temp);
		}
		else{
			ReplayBatch temp = {arg_state, arg_action, arg_reward};
			this->memory.pop();
			this->memory.push(temp);
		}
	}
};

//NeuralNetworks

class Linear{
private:
	vector<vector<double>> w;
	vector<double> b;
	vector<Tensor> gradint_chain;
	int in_dim;
	int out_dim;
public:
	Linear(int in_features, int out_features){
		in_dim  = in_features;
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

//

class DQN{
public:
	Linear layer1 = Linear(2, 3);
	Linear layer2 = Linear(3, 3);
	Linear layer3 = Linear(3, 1);
	vector<gradient> gradient_tape;
	
	DQN(){
	}

	vector<double> forward(vector<double> state){
		vector<double> x;
		x = relu(layer1.forward(state));
		x = relu(layer2.forward(x));
		x = sigmoid(layer3.forward(x));
		return x;
	}

	void summary(){
		return;
	}

	~DQN(){
	}
};

//tic tac toe

class TTT{
private:
	vector<vector<double>> game_screen;
	int width;
	int height;
	int id;
public:
	TTT(){
		id     = 1;
		width  = 3;
		height = 3;
		/*
		 * {{0, 0, 0},
		 *  {0, 0, 0},
		 *  {0, 0, 0}}
		 */
		for(int w = 0; w < width; w++){
			vector<double> temp;
			for(int h = 0; h < height; h++){
				temp.push_back(0);
			}
			game_screen.push_back(temp);
		}
	}
	int place_and_update(int x, int y){
		if(game_screen[y][x] != 0) return -1;	//unable move
		else{
			game_screen[y][x] = id;
			id *= -1;	
		}
		return 0;
	}
	double reward(){	//checks if game is over and rewards current agent
		double counts = 0;
		for(vector<double> row : game_screen){
			for(double unit : row){
				counts += unit;
			}
			if(counts == id * (-3)){
				this->reset_and_restart();
				return 1;      //win
			}
			else if(counts == id * 3){
				this->reset_and_restart();
				return -1;     //lose
			}
			else{
				counts = 0;
			}
		}
		for(int column = 0; column < width; column++){
			for(vector<double> row : game_screen){
				counts += row[column];
			}
			if(counts == id * (-3)){
				this->reset_and_restart();
				return 1;     //win
			}
			if(counts == id * 3){
				this->reset_and_restart();
				return -1;    //lose
			}
			else{
				counts = 0;
			}
		}
		for(int p = 0; p < width; p++){
			counts += game_screen[p][p];
		}
		if(counts == id * (-3)){
			this->reset_and_restart();
			return 1;
		}
		if(counts == id * 3){
			this->reset_and_restart();
			return -1;
		}
		else{
			counts = 0;
		}
		
		for(int p = 0; p < width; p++){
			counts += game_screen[p][2 - p];
		}
		if(counts == id * (-3)){
			this->reset_and_restart();
			return 1;
		}
		if(counts == id * 3){
			this->reset_and_restart();
			return -1;
		}
		else{
			counts = 0;
		}
		return 0;	//no one won::game continues
	}

	void reset_and_restart(){
		id = 1;
		/*
		 * {{0, 0, 0},
		 *  {0, 0, 0},
		 *  {0, 0, 0}}
		 */
		for(int w = 0; w < width; w++){
			vector<double> temp;
			for(int h = 0; h < height; h++){
				temp.push_back(0);
			}
			game_screen.push_back(temp);
		}
	}

	vector<double> get_screen(){
		vector<double> OUT;
		for(vector<double> row : this->game_screen){
			for(double column : row){
				OUT.push_back(column);
			}
		}
		return OUT;
	}

	void print_game_screen(){
		cout << "----------\n";
		for(vector<double> row : game_screen){
			for(double column : row){
				if(column == 1) cout << "O";
				else if (column == -1) cout << "X";
				else cout << "0";
			}
			cout << "\n";
		}
		cout << endl;
	}
};

void print_vector(vector<double> arg_vec){	//obsolete func.
	for(double unit : arg_vec){
		cout << unit << ",";
	}
	cout << endl;
}

int main(){
	vector<vector<double>> x = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
	vector<double> y = {0.0,1.0,1.0,0.0};
	DQN net;
	print_vector(net.forward(x[0]));
	return 0;
}
