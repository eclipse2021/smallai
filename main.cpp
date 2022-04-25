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

//Layers

class Linear{
private:
	vector<double> IN;
	vector<double> OUT;
	int in_dim;
	int out_dim;
public:
	vector<vector<double>> w;
	vector<double> b;

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
			IN.push_back(0.0);
		}
		for(int i = 0; i < out_features; i++){
			OUT.push_back(0.0);
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
	
	vector<double> *forward(vector<double> x, vector<grad> *argp_tape){
		IN = x;
		for(int i = 0; i < out_dim; i++){
			for(int o = 0; o < in_dim; o++){
				OUT[i] += w[i][o] * IN[o];

				grad dw{IN[o], &w[i][o], &OUT[i]};
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

				grad dw{(*x)[o], &w[i][o], &OUT[i]};
				grad din{w[i][o], &(*x)[o], &OUT[i]};
				argp_tape->push_back(dw);
				argp_tape->push_back(din);
			}
			OUT[i] += b[i];

			grad db{1.0, &b[i], &OUT[i]};
			argp_tape->push_back(db);
		}
		return &OUT;
	}

	void backward(vector<grad> *argp_tape){
	}

	int capacity(){
		/*  reutrns the number of total parameters  */
		return ((this->in_dim * this->out_dim) + this->out_dim);
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
			OUT.push_back(0.0);
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
};

class Sigmoid{
private:
	vector<double> OUT;
	int dim;
public:
	Sigmoid(int arg_dim){
		dim = arg_dim;
		for(int i = 0; i < dim; i++){
			OUT.push_back(0.0);
		}
	}

	vector<double> *forward(vector<double> *x, vector<grad> *argp_tape){
		for(int i = 0; i < dim; i++){
			OUT[i] = 1/(1 + exp(-1 * (*x)[i]));

			grad dydx{OUT[i] * (1 - OUT[i]), &(*x)[i], &OUT[i]};
			argp_tape->push_back(dydx);
		}

		return &OUT;
	}

	~Sigmoid(){}
};
//

class DQN{
public:
	Linear fc1  = Linear(2, 3);
	Relu relu1  = Relu(3);
	Linear fc2  = Linear(3, 3);
	Relu relu2  = Relu(3);
	Linear fc3  = Linear(3, 1);
	Sigmoid out = Sigmoid(1);

	vector<grad> gradient_tape;
	
	DQN(){
	}

	vector<double> *forward(vector<double> state){
		cout << "[debug/DQN/*forward()]entry" << endl; // passed
		vector<double> *x;
		x = fc1.forward(state, &gradient_tape);
		x = relu1.forward(x, &gradient_tape);
		x = fc2.forward(x, &gradient_tape);
		x = relu2.forward(x, &gradient_tape);
		x = fc3.forward(x, &gradient_tape);
		x = out.forward(x, &gradient_tape);
		cout << "[debug/DQN/*forward()]returned *" << endl; // passed
		return x;
	}

	void backward(){
		cout << "[degug]tracing gradients..." << endl;    // debug log
		vector<grad> gradient_per_parameter;    //
		vector<grad> delta_parameter;    //actual gradient

		for(grad gr : this->gradient_tape){
			if(gr.y == this->gradient_tape.back().y && gr.self != this->gradient_tape.back().self){
				gradient_per_parameter.push_back(gr);
			}
		}
		gradient_per_parameter.push_back(this->gradient_tape.back());
		for(grad chain : gradient_per_parameter){
			int require_gradient = 1;
			for(grad gradient : this->gradient_tape){
				if(chain.self == gradient.y){
					require_gradient = 0;
					grad new_chain{chain.value * gradient.value, gradient.self, chain.y};
					gradient_per_parameter.push_back(new_chain);
				}
			}
			if(require_gradient == 0){
				int idx = 0;
				while (idx < gradient_per_parameter.size()){
					if(gradient_per_parameter[idx].self == chain.self){
						gradient_per_parameter.erase(gradient_per_parameter.begin() + idx);
					}
					idx ++;
				}
			}
		}

		int DEBUG_parameter_counter = 0;    //debug
		for(grad gr : gradient_per_parameter){
			int is_already_set = 0;
			for(grad delta : delta_parameter){
				if(delta.self == gr.self){
					int is_already_set = 1;
					delta.value += gr.value;
				}
			}
			if(is_already_set == 0){
				delta_parameter.push_back(gr);
				DEBUG_parameter_counter ++;    //debug
			}
		}
		cout << "[debug]" << "fc1:" << fc1.capacity() << "parameters online\n";    // debug
		cout << "[debug]" << "fc2:" << fc2.capacity() << "parameters online\n";    // debug
		cout << "[debug]" << "fc3:" << fc3.capacity() << "parameters online\n";    // debug

		cout << "[debug]found" << DEBUG_parameter_counter << "parameters to update\n";    // debug

		/*
		cout << "[debug]updating...\n";
		for(grad delta : delta_parameter){
			//update parameters
			*(delta.self) -= delta.value;
		}
		*/
		cout << "[debug]...all parameters updated" << endl;    // debug log
	}

	void zero_grad(){
		this->gradient_tape.clear();
	}

	void summary(){
	}

	~DQN(){
	}
};

//MSE
/*
double loss(double *prediction, double *lable, vector<grad> *argp_tape){
	double loss = (*prediction) - (*lable);
	double squared_loss = loss * loss;
	grad dE_dy{loss, prediction, lable};
	argp_tape->push_back(dE_dy);
	return squared_loss/2;
}
*/
double loss(vector<double> *prediction, double *lable, vector<grad> *argp_tape){
	cout << "[loss()] entry" << endl;
	int idx = 0;
	double total_loss = 0;
	while(idx < (*prediction).size()){
		double loss = (*prediction)[idx] - *lable;
		double squared_loss = loss * loss;
		grad dE_dy{loss, &(*prediction)[idx], lable};
		argp_tape->push_back(dE_dy);
		total_loss += squared_loss/2;
		idx ++;
	}
	return total_loss;
}
class MSE{
private:
public:
	MSE(){
	}
	~MSE(){
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

void print_vector(vector<grad> arg_vec){
	for(grad unit : arg_vec){
		cout << unit.value << " , ";
	}
	cout << endl;
}

int main(){
	// test data
	vector<vector<double>> x = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
	vector<double> y = {0.0,1.0,1.0,0.0};
	//
	
	DQN net;
	cout << "[debug/main]constructed network\n"; //passed
	double E = loss(net.forward(x[0]), &(y[0]), &(net.gradient_tape));
	cout << "[debug/main]forward complete\n";
	cout << "[debug/main]calculated loss" << endl;
	net.backward();

	return 0;
}
