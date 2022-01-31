#include"dar.h"

dar::dar() : m_capacity(1), m_used(0){
	m_array = new double[m_capacity];
}

dar::dar(int capacity) : m_capacity(capacity), m_size(0){
	m_array = new double[m_capacity];
}
dar::~dar(){
	delete[] m_array;
}


void dar::pop(){
	if(m_size == 0) return;

	m_size--;
	double* temp = new double[m_size];

	for(int i = 0; i < m_size; i++){
		temp[i] = m_array[i];
	}
}
