#include<iostream>
#include<vector>

using namespace std;

void app(vector<double> *argp_li){
	&argp_li.push_back(0.0);
}

int main(){
	vector<double> li;
	app(*li);
	std::cout << li[0] << std::endl;
	return 0;
}
