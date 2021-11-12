#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "_jbgrav.h"


/* interface based heavily on Dan Foreman-Mackey's example */

static char module_docstring[] =
	"Calculation of gravitational forces using C.";

static char direct_summation_docstring[] =
	"Calculate the gravitational acceleration on every particle in the snapshot from every other particle using direct summation.";

static char treeforce_docstring[] =
	"Calculate the gravitational acceleration on every particle in the snapshot from every other particle using a Barnes-Hunt tree.";


static PyMethodDef module_methods[] = {
	{"direct_summation", (PyCFunction)jbgrav_direct_summation, METH_VARARGS, direct_summation_docstring},
	{"tree_force", (PyCFunction)jbgrav_tree_force, METH_VARARGS, treeforce_docstring},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit__jbgrav(void)
{
#if PY_MAJOR_VERSION >= 3
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_jbgrav",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    m = PyModule_Create(&moduledef);
    if (!m) return NULL;

	/* Load numpy functionality */
	import_array();

    return m;    
#else
	PyObject *m = Py_InitModule3("_jbgrav", module_methods, module_docstring);
	if (m == NULL)
		return NULL;

	/* Load numpy functionality */
	import_array();
#endif
}


/* main wrapper - get arguments into a useful state, call workhorse, and return as
 * a numpy array */
static PyObject *jbgrav_direct_summation(PyObject *self, PyObject *args)
{
	PyObject *pos_obj;  /* comes in as an Nx3 np.ndarray */
	PyObject *mass_obj; /* comes in as an N-element np.ndarray */
	double eps;
	int np;

	if (!PyArg_ParseTuple(args, "OOd", &pos_obj, &mass_obj, &eps))
		return NULL;

	/* turn into numpy arrays */
	PyObject *posarray = PyArray_FROM_OTF(pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyObject *massarray = PyArray_FROM_OTF(mass_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	/* throw exception if necessary */
	if (posarray == NULL || massarray == NULL) {
		Py_XDECREF(posarray);
		Py_XDECREF(massarray);
		return NULL;
	}

	/* make sure it's Nx3 */
	if(PyArray_NDIM(posarray) != 2) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		PyErr_SetString(PyExc_RuntimeError, "Position array does not have 2 dimensions.");
		return NULL;
	}
	if( (int)PyArray_DIM(posarray, 1) != 3 ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		PyErr_SetString(PyExc_RuntimeError, "Position array is not Nx3.");
		return NULL;
	}
	np = (int)PyArray_DIM(posarray, 0);
	/* and mass array has the same number of elements */
	if( (int)PyArray_DIM(massarray, 0) != np ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		PyErr_SetString(PyExc_RuntimeError, "Mass array and position array contain different numbers of particles.");
		return NULL;
	}

	/* create an output array */
	PyObject *forcearray = PyArray_NewLikeArray(posarray, NPY_ANYORDER, NULL, 1);

	/* call the workhorse */
	if (directsummation_workhorse(posarray, massarray, np, eps, forcearray) == NULL) {
		Py_DECREF(posarray);
		Py_DECREF(forcearray);
		PyErr_SetString(PyExc_RuntimeError, "Error in direct summation C code.");
		return NULL;
	}

	/* clean up the intermediate input ndarrays */
	Py_DECREF(posarray);
	Py_DECREF(massarray);

	/* return the output */
	return forcearray;
}




/* Workhorse part here. This part is in dimensionless units, so the driver
 * function in python will have to do the conversions and make sure that
 * it's a numpy array */
PyObject* directsummation_workhorse(PyArrayObject* pos, PyArrayObject* mass, int np, double eps, PyArrayObject* forcearray)
{
	double *dpos,*invdpos3;
	double dpos2, dpos2_plus_eps2;
	double diff,diff2,eps2;
	double *forceelement;
	int i,j,k;

	dpos = malloc(sizeof(double) * np * np * 3);
	invdpos3 = malloc(sizeof(double) * np * np);
	if((dpos==NULL) || (invdpos3==NULL)) return NULL;

	eps2 = eps*eps;
	
	/* loop through arrays calculating the dpos array */
	for (i=0; i<np; i++) {
		for(j=i+1; j<np; j++) {
			dpos2 = 0.0;
			for(k=0; k<3; k++) {
			  diff = (*(double*)PyArray_GETPTR2(pos,i,k)) - (*(double*)PyArray_GETPTR2(pos,j,k));
			  diff2 = diff*diff;
			  dpos[i*np*3 + j*3 + k] = -diff;
			  dpos[j*np*3 + i*3 + k] = diff;
			  dpos2 += diff2;
			}
			dpos2_plus_eps2 = dpos2 + eps2;
			invdpos3[i*np + j] = 1.0 / dpos2_plus_eps2 / sqrt(dpos2_plus_eps2); 
			/* based on my tests, this is twice as fast as pow(x, -1.5) */
			invdpos3[j*np + i] = invdpos3[i*np + j];
			
		}
	}

	/* loop through each particle and add up forces */
	for (i=0; i<np; i++) {
		for(k=0; k<3; k++) {
			forceelement = (double*) PyArray_GETPTR2(forcearray, i, k);
			*forceelement = 0.0;
			for(j=0; j<np; j++) {
			    if (i==j) continue;  /* no self force */

				(*forceelement) += *(double*)PyArray_GETPTR1(mass,j) *
					dpos[i*np*3 + j*3 + k] * invdpos3[i*np + j];
			}
		}
	}

	/* clear up dpos and invdpos3 arrays */
	free(dpos);
	free(invdpos3);

	/* return None */
	return Py_None;
}




/* initialize a tree node */
struct gravoct_node *gravoct_init(double *center, double size) {
	int i;

	struct gravoct_node* root = (struct gravoct_node*) malloc(sizeof(struct gravoct_node));
	if(root==NULL) {exit(209);}

	root->size = size;
	root->halfsize = 0.5*size;
	for(i=0; i<3; i++) {
		root->center[i] = center[i];
		root->boxmin[i] = center[i] - root->halfsize;
		root->boxmax[i] = center[i] + root->halfsize;
		root->firstmoment[i] = 0.0;
		root->COM[i] = 0.0;
	}
	for(i=0; i<8; i++) {
		root->branches[i] = NULL;
	}
	root->mass = 0.0;
	root->empty = 1;
	root->COMvalid = 0;
	root->leaf = NULL;

	return root;
}

/* add a particle to a tree node */
void gravoct_add_particle(struct gravoct_node *tree, struct gravoct_particle *p) {
	int i,bnum;
	int subnode[3];
	double subcenter[3];

	if (tree->empty) {
		/* turn into leaf */
		tree->empty = 0;
		tree->leaf = p;
		/* update mass and COM */
		tree->mass = p->mass;
		for(i=0; i<3; i++) {
			tree->firstmoment[i] = p->mass * p->pos[i];
		}
	} else if(tree->leaf) {
		/* move leaf to a subnode */
		gravoct_calc_subnode(tree, tree->leaf, subnode);
		bnum = gravoct_calc_branchnum(subnode);
		for (i=0; i<3; i++) {
			subcenter[i] = tree->center[i] + subnode[i]*0.5*tree->halfsize;
		}
		/* create the branch and add the leaf */
		tree->branches[bnum] = gravoct_init(subcenter, tree->halfsize);
		gravoct_add_particle(tree->branches[bnum], tree->leaf);
		tree->leaf = NULL;
		/* now try re-adding the original particle, which will trigger the next case */
		gravoct_add_particle(tree, p);
		/* note that we do *not* update the node mass and COM here because it
		 * will already be done when the next case is triggered */
	} else {
		/* add to subnode */
		gravoct_calc_subnode(tree, p, subnode);
		bnum = gravoct_calc_branchnum(subnode);
		if(tree->branches[bnum]) {
			/* already exists, so add this particle to it */
			gravoct_add_particle(tree->branches[bnum], p);
		} else {
			/* create it first */
			for (i=0; i<3; i++) {
				subcenter[i] = tree->center[i] + subnode[i]*0.5*tree->halfsize;
			}
			tree->branches[bnum] = gravoct_init(subcenter, tree->halfsize);
			gravoct_add_particle(tree->branches[bnum], p);
		}
		/* update node mass and COM */
		tree->mass += p->mass;
		for(i=0; i<3; i++) {
			tree->firstmoment[i] += p->mass * p->pos[i];
		}
	}
}


/* calculate the branch number for a given octent */
int gravoct_calc_branchnum(int *subnode) {
	int i,b;
	b = 0;
	for(i=0; i<3; i++) {
		if(subnode[i] > 0) {
			b += (1 << i);
		}
	}
	return b;
}

/* calculate the octent of a particle given the center of the node */
void gravoct_calc_subnode(struct gravoct_node *tree, struct gravoct_particle *p, int *subnode) {
	int i;
	for(i=0; i<3; i++) {
		if( p->pos[i] > tree->center[i] ) {
			subnode[i] = 1;
		} else {
			subnode[i] = -1;
		}
	}
}

/* need to call this on a node before using the COM value -- do this
 * after the tree has been fully built and before using the COM value
 * in the force calculation */
void gravoct_finalize(struct gravoct_node *tree) {
	int i;

	if(!(tree->COMvalid)) {
		/* take it from particle pos if leaf, otherwise calculate from first moment */
		if(tree->leaf) {
			for(i=0; i<3; i++) {
				tree->COM[i] = tree->leaf->pos[i];
			}
		} else {
			for(i=0; i<3; i++) {
				tree->COM[i] = tree->firstmoment[i] / tree->mass;
			}
		}
		tree->COMvalid = 1;
	}
}

/* walk the tree to calculate the acceleration at position pos from the tree tree, and
 * put the result in force */
void gravoct_calc_accel(struct gravoct_node *tree, double *pos, double eps, double theta, double *force)
{
	int i,j;
	double node_dist, d_pos[3], invdpos3, dpos2, diff, diff2, eps2;
	double dpos2_plus_eps2;
	double branchforce[3];

	eps2 = eps*eps;

	node_dist = 0.0;
	for(i=0; i<3; i++) {
		node_dist += (tree->center[i] - pos[i]) * (tree->center[i] - pos[i]);
	}
	node_dist = sqrt(node_dist);
	/* check opening criterion */
	if( (tree->leaf) || ((tree->size / node_dist) < theta) ) {
		/* either a leaf or it is distant enough that it can be approximated
		 *  by its COM. Either way, totol node properties are sufficient */
		gravoct_finalize(tree);
		dpos2 = 0.0;
		for(i=0; i<3; i++) {
			diff = tree->COM[i] - pos[i];
			diff2 = diff*diff;
			d_pos[i] = diff;
			dpos2 += diff2;
		}
		dpos2_plus_eps2 = dpos2 + eps2;
		invdpos3 = 1.0 / dpos2_plus_eps2 / sqrt(dpos2_plus_eps2); /* 2x faster than pow(dpos2 + eps2, -1.5); */

		for(i=0; i<3; i++) {
			force[i] = d_pos[i] * tree->mass * invdpos3;
		}
	} else {
		/* needs to be opened */
		for(i=0; i<3; i++) {
			force[i] = 0.0;
		}
		for(j=0; j<8; j++) {
			if(tree->branches[j]) {
				gravoct_calc_accel(tree->branches[j], pos, eps, theta, branchforce);
				for(i=0; i<3; i++) {
					force[i] += branchforce[i];
				}
			}
		}
	}

	return;
}


void gravoct_deltree(struct gravoct_node *tree) {
	int j;

	/* first destroy branches */
	for(j=0; j<8; j++) {
		if(tree->branches[j]) {
			gravoct_deltree(tree->branches[j]);
		}
	}
	/* and this one */
	if(tree->leaf) {
		free(tree->leaf);
	}
	free(tree);
}



/* main wrapper - get arguments into a useful state, call workhorse, and return as
 * a numpy array */
static PyObject *jbgrav_tree_force(PyObject *self, PyObject *args)
{
	PyObject *pos_obj;  /* comes in as an Nx3 np.ndarray */
	PyObject *mass_obj; /* comes in as an N-element np.ndarray */
	double eps;
	int np;

	if (!PyArg_ParseTuple(args, "OOd", &pos_obj, &mass_obj, &eps))
		return NULL;

	/* turn into numpy arrays */
	PyObject *posarray = PyArray_FROM_OTF(pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyObject *massarray = PyArray_FROM_OTF(mass_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	/* throw exception if necessary */
	if (posarray == NULL || massarray == NULL) {
		Py_XDECREF(posarray);
		Py_XDECREF(massarray);
		return NULL;
	}

	/* make sure it's Nx3 */
	if(PyArray_NDIM(posarray) != 2) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		PyErr_SetString(PyExc_RuntimeError, "Position array does not have 2 dimensions.");
		return NULL;
	}
	if( (int)PyArray_DIM(posarray, 1) != 3 ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		PyErr_SetString(PyExc_RuntimeError, "Position array is not Nx3.");
		return NULL;
	}
	np = (int)PyArray_DIM(posarray, 0);
	/* and mass array has the same number of elements */
	if( (int)PyArray_DIM(massarray, 0) != np ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		PyErr_SetString(PyExc_RuntimeError, "Mass array and position array contain different numbers of particles.");
		return NULL;
	}

	/* create an output array */
	PyObject *forcearray = PyArray_NewLikeArray(posarray, NPY_ANYORDER, NULL, 1);

	/* call the workhorse */
	if (treeforce_workhorse(posarray, massarray, np, eps, forcearray) == NULL) {
		Py_DECREF(posarray);
		Py_DECREF(forcearray);
		PyErr_SetString(PyExc_RuntimeError, "Error in tree C code.");
		return NULL;
	}

	/* clean up the intermediate input ndarrays */
	Py_DECREF(posarray);
	Py_DECREF(massarray);

	/* return the output */
	return forcearray;
}

/* Tree workhorse part here. This part is in dimensionless units, so the driver
 * function in python will have to do the conversions and make sure that
 * it's a numpy array */
PyObject* treeforce_workhorse(PyArrayObject* pos, PyArrayObject* mass, int np, double eps, PyArrayObject* forcearray)
{
	struct gravoct_node *root;
	struct gravoct_particle *p;
	double min[3], max[3], boxsize, boxcenter[3],q;
	double thisforce[3],thispos[3];

	double theta=0.7;

	int i,j;

	/* get basic tree parameters */
	for(i=0; i<3; i++) {
		min[i] = *(double *)PyArray_GETPTR2(pos,0,i);
		max[i] = min[i];
	}
	for(i=1; i<np; i++) {
		for(j=0; j<3; j++) {
			q = *(double *)PyArray_GETPTR2(pos,i,j);
			if(q < min[j]) {
				/* update minimum box coords */
				min[j] = q;
			}
			if(q > max[j]) {
				/* update maximum box coords */
				max[j] = q;
			}
		}
	}
	boxsize = max[0]-min[0] + eps;
	for(i=1; i<3; i++) {
		if( (max[i]-min[i]) > boxsize ) {
			boxsize = max[i]-min[i] + eps;
		}
	}
	for(i=0; i<3; i++) {
		boxcenter[i] = 0.5*(min[i] + max[i]);
	}

	/* build the tree */
	root = gravoct_init(boxcenter, boxsize);
	for(i=0; i<np; i++) {
		/* create a particle */
		p = (struct gravoct_particle *)malloc(sizeof(struct gravoct_particle));
		if(p==NULL) {exit(435);}
		p->mass = *(double *)PyArray_GETPTR1(mass,i);
		for(j=0; j<3; j++) {
			p->pos[j] = *(double *)PyArray_GETPTR2(pos,i,j);
		}
		/* and add it */
		gravoct_add_particle(root, p);
	}

	/* calculate forces */
	for(i=0; i<np; i++) {
		for(j=0; j<3; j++) {
			thispos[j] = *(double *)PyArray_GETPTR2(pos,i,j);
		}
		gravoct_calc_accel(root, thispos, eps, theta, thisforce);
		/* save in forcearray */
		for(j=0; j<3; j++) {
			*(double *)PyArray_GETPTR2(forcearray,i,j) = thisforce[j];
		}
	}

	/* destroy tree */
	gravoct_deltree(root);


	/* return None */
	return Py_None;
}

