class dar{
private:
	int* m_array;
	int m_capacity;
	int m_size;
public:
	dar();
	dar(int size);
	~dar();
	
	void pop();
	void push(double data);
	void print();
	int &operator[](int index);
};
