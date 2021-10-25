static PyObject *jbgrav_direct_summation(PyObject *self, PyObject *args);
PyObject* directsummation_workhorse(PyArrayObject* pos, PyArrayObject* mass, int np, double eps, PyArrayObject* forcearray);


struct gravoct_particle {
	double pos[3];
	double mass;
};



struct gravoct_node {
	double center[3];
	double size,halfsize,boxmin[3],boxmax[3],mass;
	double firstmoment[3],COM[3];
	struct gravoct_node* branches[8];
	char empty,COMvalid;
	struct gravoct_particle *leaf;
};

struct gravoct_node *gravoct_init(double *center, double size);
void gravoct_add_particle(struct gravoct_node *tree, struct gravoct_particle *p);
int gravoct_calc_branchnum(int *subnode);
void gravoct_calc_subnode(struct gravoct_node *tree, struct gravoct_particle *p, int *subnode);
void gravoct_finalize(struct gravoct_node *tree);
void gravoct_deltree(struct gravoct_node *tree);
static PyObject *jbgrav_tree_force(PyObject *self, PyObject *args);
PyObject* treeforce_workhorse(PyArrayObject* pos, PyArrayObject* mass, int np, double eps, PyArrayObject* forcearray);


