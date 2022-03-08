#include<iostream>
#include<vector>

using namespace std;

class Temtem
{
private:
	double value;
public:
	Temtem()
	{
		value = 0.0;
	}
	double rt_pointer()
	{
		return &value;
	}
}

void print_value(double *argp_double)
{
	cout << argp_double;
}

int main()
{
	Temtem tem;
	print_value(tem.value);

	return 0;
}
