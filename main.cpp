#include<iostream>

#include<vector>
#include<queue>

#include<random>

using namespace std;

//gradient

typedef struct{
	double value;
	double *self;
	double *y;
}grad;

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
	vector<double> IN;
	vector<double> OUT;
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
		for(int i = 0; i < in_features; i++){
			IN[i] = 0.0;
		}
		for(int i = 0; i < out_features; o++){
			OUT[i] = 0.0;
		}
	}

	void print_weight(){
		cout << "w = {" << "\n";+
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
	
	vector<double> *forward(vector<double> x, vector<grad> *argp_tape){
		IN = x;
		for(int i = 0; i < out_dim; i++){
			for(int o = 0; o < in_dim; o++){
				OUT[i] += w[i][o] * IN[o];

				gard dw{IN[o], &w[i][o], &OUT[i]};
				grad din{w[i][o], &IN[o], &OUT[i]};
				argp_tape->push_back(dw);
				argp_tape->push_back(din);
			}
			OUT[i] += b[i];

			grad db{1.0, &b[i], &OUT[i]};
			argp_tape->push_back(db);
		}
		return &OUT;
	}

	vector<double> *forward(vector<double> *x, vector<grad> *argp_tape){
		for(int i = 0; i < out_dim; i++){
			for(int o = 0; o < in_dim; o++){
				OUT[i] += w[i][o] * (*x)[o];

				gard dw{&(*x)[o]/* */, &w[i][o], &OUT[i]};
				grad din{w[i][o], &(*x)[o]/* */, &OUT[i]};
				argp_tape->push_back(dw);
				argp_tape->push_back(din);
			}
			OUT[i] += b[i];

			grad db{1.0, &b[i], &OUT[i]};
			argp_tape->push_back(db);
		}
		return &OUT;
	}

	~Linear(){
	}
};


class Relu{
private:
	vector<double> OUT;
	int dim;
public:
	Relu(int arg_dim){
		dim = arg_dim;
		for(int i = 0; i < dim; i++){
			OUT[i] = 0.0;
		}
	}

	vector<double> *forward(vector<double> *x, vector<grad> *argp_tape){
		for(int i = 0; i < dim;  i++){
			if((*x)[i] <= 0){
				OUT[i] = 0;
				grad dydx{0, &(*x)[i], &OUT[i]};
				argp_tape->push_back(dydx);
			}
			else{
				OUT[i] = (*x)[i];
				grad dydx{1, &(*x)[i], &OUT[i]};
				argp_tape->push_back(dydx);
			}
		}

		return &OUT;
	}

	~Relu(){
	}
}

class Sigmoid{
private:
	vector<double> OUT;
	int dim;
public:
	Sigmoid(int arg_dim){
		dim = arg_dim;
		for(int i = 0; i < dim; i++){
			IN[i] = 0.0;
			OUT[i] = 0.0;
		}
	}

	vector<double> *forward(vector<double> *x, vector<grad> *argp_tape){
		for(int i = 0; i < dim; i++){
			OUT[i] = 1/(1 + exp(-1 * (*x)[i]);

			grad dydx{OUT[i] * (1 - OUT[i]), &(*x)[i], &OUT[i]};
			argp_tape->push_back(dydx);
		}
	}

	~Sigmoid(){}
};
//

class DQN{
public:
	Linear fc1 = Linear(2, 3);
	Relu relu1 = Relu(3);
	Linear fc2 = Linear(3, 3);
	Relu relu2 = Relu(3);
	Linear fc3 = Linear(3, 1);
	Sigmoid out = Sigmoid(1);
	vector<grad> gradient_tape;
	vector<grad> *gradient;
	
	DQN(){
		gradient = &gradient_tape;
	}

	vector<double> forward(vector<double> state){
		vector<double> *x;
		x = fc1.forward(state, gradient);
		x = relu1(x, gradient);
		x = fc2.forward(x, gradient);
		x = relu(x, gradient);
		x = layer3.forward(x, gradient);
		x = sigmoid(x, gradient;
		return *x;
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
	// test data
	vector<vector<double>> x = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
	vector<double> y = {0.0,1.0,1.0,0.0};
	//
	
	DQN net;
	print_vector(net.forward(x[0]));
	return 0;
}
