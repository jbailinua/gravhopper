#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "_jbgrav.h"


/* interface based heavily on Dan Foreman-Mackey's example */

static char module_docstring[] =
	"Calculation of gravitational forces using C.";

static char direct_summation_docstring[] =
	"Calculate the gravitational acceleration or potential on every particle in the snapshot from every other particle using direct summation.";

static char direct_summation_position_docstring[] =
	"Calculate the gravitational acceleration or potential at a set of positions from every particle in a simulation using direct summation.";

static char treeforce_docstring[] =
	"Calculate the gravitational acceleration or potential on every particle in the snapshot from every other particle using a Barnes-Hut tree.";

static char treeforce_position_docstring[] =
	"Calculate the gravitational acceleration or potential at a set of positions from every particle in a simulation using a Barnes-Hut tree.";


static PyMethodDef module_methods[] = {
	{"direct_summation", (PyCFunction)jbgrav_direct_summation, METH_VARARGS, direct_summation_docstring},
	{"direct_summation_position", (PyCFunction)jbgrav_direct_summation_position, METH_VARARGS, direct_summation_position_docstring},
	{"tree_force", (PyCFunction)jbgrav_tree_force, METH_VARARGS, treeforce_docstring},
	{"tree_force_position", (PyCFunction)jbgrav_tree_force_position, METH_VARARGS, treeforce_position_docstring},
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
	int np, calc_force, calc_potential;

	if (!PyArg_ParseTuple(args, "OOdpp", &pos_obj, &mass_obj, &eps, &calc_force, &calc_potential))
		return NULL;

	/* turn into numpy arrays */
	PyArrayObject *posarray = (PyArrayObject*) PyArray_FROM_OTF(pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *massarray = (PyArrayObject*) PyArray_FROM_OTF(mass_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
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

	/* create output arrays */
	PyArrayObject *forcearray, *potarray;
	if(calc_force) {
        forcearray = (PyArrayObject*) PyArray_NewLikeArray(posarray, NPY_ANYORDER, NULL, 1);
        /* throw exception if necessary */
        if (forcearray == NULL) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_XDECREF(forcearray);
            return NULL;
        }
    } else {
        forcearray = NULL;
    }
    if(calc_potential) {
        potarray = (PyArrayObject*) PyArray_NewLikeArray(massarray, NPY_ANYORDER, NULL, 1);
        /* throw exception if necessary */
        if (potarray == NULL) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_XDECREF(forcearray);
            Py_XDECREF(potarray);
            return NULL;
        }
    } else {
        potarray = NULL;
    }

	/* call the workhorse */
	if (directsummation_workhorse(posarray, massarray, np, eps, calc_force, calc_potential, forcearray, potarray) == NULL) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_XDECREF(forcearray);
		Py_XDECREF(potarray);
		PyErr_SetString(PyExc_RuntimeError, "Error in direct summation C code.");
		return NULL;
	}

	/* clean up the intermediate input ndarrays */
	Py_DECREF(posarray);
	Py_DECREF(massarray);

	/* return the output */
	if(calc_force) {
	    if(calc_potential) {
	        /* create and return tuple with acceleration and energy */
	        PyObject *forcepot_tuple = PyTuple_Pack(2, forcearray, potarray);
	        return forcepot_tuple;
	    } else {
	        /* only return acceleration */
	        return (PyObject*) forcearray;
	    }
	} else {
	    if(calc_potential) {
	        /* only return potential energy */
	        return (PyObject*) potarray;
	    } else {
	        /* this should never run, but if so return None */
	        return Py_None;
	    }
	}
}

/* Workhorse part here. This part is in dimensionless units, so the driver
 * function in python will have to do the conversions and make sure that
 * it's a numpy array */
PyObject* directsummation_workhorse(PyArrayObject* pos, PyArrayObject* mass, int np, double eps, int calc_force, int calc_potential, PyArrayObject* forcearray, PyArrayObject* potarray)
{
	double *dpos,*invdpos3,*invd;
	double dpos2, dpos2_plus_eps2, inv_sqrt_dpos2_plus_eps2;
	double diff,diff2,eps2;
	double *forceelement,*potelement;
	int i,j,k;

	if(calc_force) {
    	dpos = malloc(sizeof(double) * np * np * 3);
	    invdpos3 = malloc(sizeof(double) * np * np);
	    if((dpos==NULL) || (invdpos3==NULL)) return NULL;
	}
	if(calc_potential) {
	    invd = malloc(sizeof(double) * np * np);
	    if(invd==NULL) return NULL;
	}

	eps2 = eps*eps;
	
	/* loop through arrays calculating the dpos and distance arrays */
	for (i=0; i<np; i++) {
		for(j=i+1; j<np; j++) {
			dpos2 = 0.0;
			for(k=0; k<3; k++) {
			  diff = (*(double*)PyArray_GETPTR2(pos,i,k)) - (*(double*)PyArray_GETPTR2(pos,j,k));
			  diff2 = diff*diff;
			  dpos2 += diff2;
			  if(calc_force) {
                  dpos[i*np*3 + j*3 + k] = -diff;
                  dpos[j*np*3 + i*3 + k] = diff;
              }
			}
			dpos2_plus_eps2 = dpos2 + eps2;
			inv_sqrt_dpos2_plus_eps2 = 1.0 / sqrt(dpos2_plus_eps2);
			if(calc_force) {
                invdpos3[i*np + j] = inv_sqrt_dpos2_plus_eps2 / dpos2_plus_eps2;
                /* based on my tests, this is twice as fast as pow(x, -1.5) */
                invdpos3[j*np + i] = invdpos3[i*np + j];
            }
            if(calc_potential) {
                invd[i*np + j] = inv_sqrt_dpos2_plus_eps2;
                invd[j*np + i] = inv_sqrt_dpos2_plus_eps2;
            }
			
		}
	}

	/* loop through each particle and add up forces/potentials */
	for (i=0; i<np; i++) {
	    if(calc_potential) {
	        potelement = (double*) PyArray_GETPTR1(potarray, i);
	        *potelement = 0.0;
	        for(j=0; j<np; j++) {
	            if (i==j) continue;  /* no self potential */
	            (*potelement) -= *(double*)PyArray_GETPTR1(mass,j) * invd[i*np + j];
	        }
	    }
	    
	    if(calc_force) {
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
	}

	/* clear up dpos, invdpos3, invd arrays */
	if(calc_force) {
        free(dpos);
        free(invdpos3);
    }
    if(calc_potential) {
        free(invd);
    }

	/* return None */
	return Py_None;
}




/* main wrapper - get arguments into a useful state, call workhorse, and return as
 * a numpy array */
static PyObject *jbgrav_direct_summation_position(PyObject *self, PyObject *args)
{
	PyObject *pos_obj;  /* comes in as an Npx3 np.ndarray */
	PyObject *mass_obj; /* comes in as an Np-element np.ndarray */
	PyObject *force_pos_obj;  /* comes in as an Nx3 np.ndarray */
	double eps;
	int np, nf;    // Number of particles, number of force locations
	int calc_force, calc_potential;

	if (!PyArg_ParseTuple(args, "OOOdpp", &pos_obj, &mass_obj, &force_pos_obj, &eps, &calc_force, &calc_potential))
		return NULL;

	/* turn into numpy arrays */
	PyArrayObject *posarray = (PyArrayObject*) PyArray_FROM_OTF(pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *massarray = (PyArrayObject*) PyArray_FROM_OTF(mass_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *forceposarray = (PyArrayObject*) PyArray_FROM_OTF(force_pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	/* throw exception if necessary */
	if (posarray == NULL || massarray == NULL || forceposarray == NULL) {
		Py_XDECREF(posarray);
		Py_XDECREF(massarray);
		Py_XDECREF(forceposarray);
		return NULL;
	}

	/* make sure particle position array is Nx3 */
	if(PyArray_NDIM(posarray) != 2) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		PyErr_SetString(PyExc_RuntimeError, "Particle position array does not have 2 dimensions.");
		return NULL;
	}
	if( (int)PyArray_DIM(posarray, 1) != 3 ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		PyErr_SetString(PyExc_RuntimeError, "Particle position array is not Nx3.");
		return NULL;
	}
	np = (int)PyArray_DIM(posarray, 0);
	/* and mass array has the same number of elements */
	if( (int)PyArray_DIM(massarray, 0) != np ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		PyErr_SetString(PyExc_RuntimeError, "Mass array and particle position array contain different numbers of particles.");
		return NULL;
	}

	/* make sure force position array is Nx3 */
	if(PyArray_NDIM(forceposarray) != 2) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		PyErr_SetString(PyExc_RuntimeError, "Force position array does not have 2 dimensions.");
		return NULL;
	}
	if( (int)PyArray_DIM(forceposarray, 1) != 3 ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		PyErr_SetString(PyExc_RuntimeError, "Force position array is not Nx3.");
		return NULL;
	}
	nf = (int)PyArray_DIM(forceposarray, 0);
	

	/* create output arrays */
	PyArrayObject *forcearray, *potarray;
	if(calc_force) {
	    forcearray = (PyArrayObject*) PyArray_NewLikeArray(forceposarray, NPY_ANYORDER, NULL, 1);
        /* throw exception if necessary */
        if (forcearray == NULL) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_DECREF(forceposarray);
            Py_XDECREF(forcearray);
            return NULL;
        }
    } else {
        forcearray = NULL;
    }
    if(calc_potential) {
        npy_intp outdims_pot[1];
        outdims_pot[0] = (npy_intp) nf;
        potarray = (PyArrayObject*) PyArray_EMPTY(1, outdims_pot, NPY_DOUBLE, 0);

        /* throw exception if necessary */
        if (potarray == NULL) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_DECREF(forceposarray);
            Py_XDECREF(forcearray);
            Py_XDECREF(potarray);
            return NULL;
        }
    } else {
        potarray = NULL;
    }
        
	
	/* call the workhorse */
	if (directsummation_position_workhorse(posarray, massarray, np, forceposarray, nf, eps, calc_force, calc_potential, forcearray, potarray) == NULL) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		Py_XDECREF(forcearray);
		Py_XDECREF(potarray);
		PyErr_SetString(PyExc_RuntimeError, "Error in direct summation position C code.");
		return NULL;
	}

	/* clean up the intermediate input ndarrays */
	Py_DECREF(posarray);
	Py_DECREF(massarray);
    Py_DECREF(forceposarray);

	/* return the output */
	if(calc_force) {
	    if(calc_potential) {
	        /* create and return tuple with acceleration and energy */
	        PyObject *forcepot_tuple = PyTuple_Pack(2, forcearray, potarray);
	        return forcepot_tuple;
	    } else {
	        /* only return acceleration */
	        return (PyObject*) forcearray;
	    }
	} else {
	    if(calc_potential) {
	        /* only return potential energy */
	        return (PyObject*) potarray;
	    } else {
	        /* this should never run, but if so return None */
	        return Py_None;
	    }
	}
}

/* Workhorse part here. This part is in dimensionless units, so the driver
 * function in python will have to do the conversions and make sure that
 * it's a numpy array.
 * This is a separate workhorse from directsummation_workhorse because the symmetry
 * there means it can do half as much work. */
PyObject* directsummation_position_workhorse(PyArrayObject* pos, PyArrayObject* mass, int np, PyArrayObject* forcepos, int nf, double eps, int calc_force, int calc_potential, PyArrayObject* forcearray, PyArrayObject* potarray)
{
	double *dpos,*invdpos3,*invd;
	double dpos2, dpos2_plus_eps2, inv_sqrt_dpos2_plus_eps2;
	double diff,diff2,eps2;
	double *forceelement, *potelement;
	int i,j,k;

    if(calc_force) {
        dpos = malloc(sizeof(double) * nf * np * 3);
        invdpos3 = malloc(sizeof(double) * nf * np);
        if((dpos==NULL) || (invdpos3==NULL)) return NULL;
    }
    if(calc_potential) {
        invd = malloc(sizeof(double) * nf * np);
        if(invd==NULL) return NULL;
    }

	eps2 = eps*eps;
	
	/* loop through arrays calculating the dpos and distance arrays */
	for (i=0; i<nf; i++) {
		for(j=0; j<np; j++) {
			dpos2 = 0.0;
			for(k=0; k<3; k++) {
			  diff = (*(double*)PyArray_GETPTR2(forcepos,i,k)) - (*(double*)PyArray_GETPTR2(pos,j,k));
			  diff2 = diff*diff;
			  dpos2 += diff2;
			  if(calc_force) {
    			  dpos[i*np*3 + j*3 + k] = -diff;
    		  }
			}
			dpos2_plus_eps2 = dpos2 + eps2;
            /* If the requested location exactly matches a particle and eps==0,
            we can get a divide by zero. In this case, that's calculating the self-force
            of a particle on itself, which is zero. */
            if(dpos2_plus_eps2==0.0) {
                if(calc_force) {
                    invdpos3[i*np + j] = 0.0;
                }
                if(calc_potential) {
                    invd[i*np + j] = 0.0;
                }
            } else {
    			inv_sqrt_dpos2_plus_eps2 = 1.0 / sqrt(dpos2_plus_eps2);
	    		if(calc_force) {
                    invdpos3[i*np + j] = inv_sqrt_dpos2_plus_eps2 / dpos2_plus_eps2;
                    /* based on my tests, this is twice as fast as pow(x, -1.5) */			
                }
                if(calc_potential) {
                    invd[i*np + j] = inv_sqrt_dpos2_plus_eps2;
                }
            }
		}
	}

	/* loop through each position and add up forces/potentials */
	for (i=0; i<nf; i++) {
	    if(calc_potential) {
	        potelement = (double*) PyArray_GETPTR1(potarray, i);
	        *potelement = 0.0;
	        for(j=0; j<np; j++) {
	            (*potelement) -= *(double*)PyArray_GETPTR1(mass,j) * invd[i*np + j];
	        }
	    }
	
	    if(calc_force) {
            for(k=0; k<3; k++) {
                forceelement = (double*) PyArray_GETPTR2(forcearray, i, k);
                *forceelement = 0.0;
                for(j=0; j<np; j++) {
                    (*forceelement) += *(double*)PyArray_GETPTR1(mass,j) *
                        dpos[i*np*3 + j*3 + k] * invdpos3[i*np + j];
                }
            }
        }
	}

	/* clear up dpos, invdpos3, invd arrays */
	if(calc_force) {
        free(dpos);
        free(invdpos3);
    }
    if(calc_potential) {
        free(invd);
    }

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
void gravoct_calc_accel(struct gravoct_node *tree, double *pos, double eps, double theta, int calc_force, int calc_potential, double *force, double *pot)
{
	int i,j;
	double node_dist, d_pos[3], invdpos3, invd, dpos2, diff, diff2, eps2;
	double dpos2_plus_eps2, inv_sqrt_dpos2_plus_eps2;
	double branchforce[3], branchpot;

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
        /* If the requested location exactly matches a leaf and eps=0,
        we can get a divide by zero. In this case, that's calculating the self-force
        of a particle on itself, which is zero. */
        if (dpos2_plus_eps2==0.0) {
            if(calc_force) {
                for(i=0; i<3; i++) {
                    force[i] = 0.0;
                }
            }
            if(calc_potential) {
                *pot = 0.0;
            }
        } else {
            inv_sqrt_dpos2_plus_eps2 = 1.0/sqrt(dpos2_plus_eps2); /* needed for both accel and pot */
            
            if(calc_force) {
                invdpos3 = inv_sqrt_dpos2_plus_eps2 / dpos2_plus_eps2; /* 2x faster than pow(dpos2 + eps2, -1.5); */
                for(i=0; i<3; i++) {
                    force[i] = d_pos[i] * tree->mass * invdpos3;
                }
            }
            if(calc_potential) {
                invd = inv_sqrt_dpos2_plus_eps2;
                *pot = - tree->mass * invd;
            }
        }
            
	} else {
		/* needs to be opened */
		if(calc_force) {
            for(i=0; i<3; i++) {
                force[i] = 0.0;
            }
        }
        if(calc_potential) {
    		*pot = 0.0;
    	}
		for(j=0; j<8; j++) {
			if(tree->branches[j]) {
				gravoct_calc_accel(tree->branches[j], pos, eps, theta, calc_force, calc_potential, branchforce, &branchpot);
				if(calc_force) {
                    for(i=0; i<3; i++) {
                        force[i] += branchforce[i];
                    }
                }
                if(calc_potential) {
    				*pot += branchpot;
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
	double eps, theta;
	int np, calc_force, calc_potential;

	if (!PyArg_ParseTuple(args, "OOddpp", &pos_obj, &mass_obj, &eps, &theta, &calc_force, &calc_potential))
		return NULL;

	/* turn into numpy arrays */
	PyArrayObject *posarray = (PyArrayObject*) PyArray_FROM_OTF(pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *massarray = (PyArrayObject*) PyArray_FROM_OTF(mass_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
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

	/* create output arrays */
	PyArrayObject *forcearray, *potarray;
	if(calc_force) {
        forcearray = (PyArrayObject*) PyArray_NewLikeArray(posarray, NPY_ANYORDER, NULL, 1);
        /* throw exception if necessary */
        if (forcearray == NULL) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_XDECREF(forcearray);
            return NULL;
        }
    } else {
        forcearray = NULL;
    }
    if(calc_potential) {
        potarray = (PyArrayObject*) PyArray_NewLikeArray(massarray, NPY_ANYORDER, NULL, 1);
        /* throw exception if necessary */
        if (potarray == NULL) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_XDECREF(forcearray);
            Py_XDECREF(potarray);
            return NULL;
        }
    } else {
        potarray = NULL;
    }

	/* call the workhorse with the particle positions as forcepos */
	if (treeforce_workhorse(posarray, massarray, np, posarray, np, eps, theta, calc_force, calc_potential, forcearray, potarray) == NULL) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_XDECREF(forcearray);
		Py_XDECREF(potarray);
		PyErr_SetString(PyExc_RuntimeError, "Error in tree C code.");
		return NULL;
	}

	/* clean up the intermediate input ndarrays */
	Py_DECREF(posarray);
	Py_DECREF(massarray);

	/* return the output */
	if(calc_force) {
	    if(calc_potential) {
	        /* create and return tuple with acceleration and energy */
	        PyObject *forcepot_tuple = PyTuple_Pack(2, forcearray, potarray);
	        return forcepot_tuple;
	    } else {
	        /* only return acceleration */
	        return (PyObject*) forcearray;
	    }
	} else {
	    if(calc_potential) {
	        /* only return potential energy */
	        return (PyObject*) potarray;
	    } else {
	        /* this should never run, but if so return None */
	        return Py_None;
	    }
	}

}

/* main wrapper - get arguments into a useful state, call workhorse, and return as
 * a numpy array */
static PyObject *jbgrav_tree_force_position(PyObject *self, PyObject *args)
{
	PyObject *pos_obj;  /* comes in as an Nx3 np.ndarray */
	PyObject *mass_obj; /* comes in as an N-element np.ndarray */
	PyObject *force_pos_obj;  /* comes in as an Nx3 np.ndarray */
	double eps, theta;
	int np, nf, calc_force, calc_potential;

	if (!PyArg_ParseTuple(args, "OOOddpp", &pos_obj, &mass_obj, &force_pos_obj, &eps, &theta, &calc_force, &calc_potential))
		return NULL; 

	/* turn into numpy arrays */
	PyArrayObject *posarray = (PyArrayObject*) PyArray_FROM_OTF(pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *massarray = (PyArrayObject*) PyArray_FROM_OTF(mass_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *forceposarray = (PyArrayObject*) PyArray_FROM_OTF(force_pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	/* throw exception if necessary */
	if (posarray == NULL || massarray == NULL) {
		Py_XDECREF(posarray);
		Py_XDECREF(massarray);
		Py_XDECREF(forceposarray);
		return NULL;
	}

	/* make sure particle position array is Nx3 */
	if(PyArray_NDIM(posarray) != 2) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		PyErr_SetString(PyExc_RuntimeError, "Particle position array does not have 2 dimensions.");
		return NULL;
	}
	if( (int)PyArray_DIM(posarray, 1) != 3 ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		PyErr_SetString(PyExc_RuntimeError, "Particle position array is not Nx3.");
		return NULL;
	}
	np = (int)PyArray_DIM(posarray, 0);
	/* and mass array has the same number of elements */
	if( (int)PyArray_DIM(massarray, 0) != np ) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
		Py_DECREF(forceposarray);
		PyErr_SetString(PyExc_RuntimeError, "Mass array and particle position array contain different numbers of particles.");
		return NULL;
	}

    /* make sure force position array is Nx3 */
    if(PyArray_NDIM(forceposarray) != 2) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_DECREF(forceposarray);
            PyErr_SetString(PyExc_RuntimeError, "Force position array does not have 2 dimensions.");
            return NULL;
    }
    if( (int)PyArray_DIM(forceposarray, 1) != 3 ) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_DECREF(forceposarray);
            PyErr_SetString(PyExc_RuntimeError, "Force position array is not Nx3.");
            return NULL;
    }
    nf = (int)PyArray_DIM(forceposarray, 0);

	/* create output arrays */
	PyArrayObject *forcearray, *potarray;
	if(calc_force) {
        forcearray = (PyArrayObject*) PyArray_NewLikeArray(forceposarray, NPY_ANYORDER, NULL, 1);
        /* throw exception if necessary */
        if (forcearray == NULL) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_DECREF(forceposarray);
            Py_XDECREF(forcearray);
            return NULL;
        }
    } else {
        forcearray = NULL;
    }
    if(calc_potential) {
        npy_intp outdims_pot[1];
        outdims_pot[0] = (npy_intp)nf;
        potarray = (PyArrayObject*) PyArray_EMPTY(1, outdims_pot, NPY_DOUBLE, 0); 

        /* throw exception if necessary */
        if (potarray == NULL) {
            Py_DECREF(posarray);
            Py_DECREF(massarray);
            Py_DECREF(forceposarray);
            Py_XDECREF(forcearray);
            Py_XDECREF(potarray);
            return NULL;
        }
    } else {
        potarray = NULL;
    }

	/* call the workhorse with the particle positions as forcepos too */
	if (treeforce_workhorse(posarray, massarray, np, forceposarray, nf, eps, theta, calc_force, calc_potential, forcearray, potarray) == NULL) {
		Py_DECREF(posarray);
		Py_DECREF(massarray);
        Py_DECREF(forceposarray);
		Py_XDECREF(forcearray);
		Py_XDECREF(potarray);
		PyErr_SetString(PyExc_RuntimeError, "Error in tree C code.");
		return NULL;
	}

	/* clean up the intermediate input ndarrays */
	Py_DECREF(posarray);
	Py_DECREF(massarray);
    Py_DECREF(forceposarray);

	/* return the output */
	if(calc_force) {
	    if(calc_potential) {
	        /* create and return tuple with acceleration and energy */
	        PyObject *forcepot_tuple = PyTuple_Pack(2, forcearray, potarray);
	        return forcepot_tuple;
	    } else {
	        /* only return acceleration */
	        return (PyObject*) forcearray;
	    }
	} else {
	    if(calc_potential) {
	        /* only return potential energy */
	        return (PyObject*) potarray;
	    } else {
	        /* this should never run, but if so return None */
	        return Py_None;
	    }
	}
}



/* Tree workhorse part here. This part is in dimensionless units, so the driver
 * function in python will have to do the conversions and make sure that
 * it's a numpy array.
 * This workhorse works for both the regular and position versions because it just
 * builds a tree based on particles and calls gravoct_calc_accel on the positions,
 * so it can just be passed a different position array or the particle one. */
PyObject* treeforce_workhorse(PyArrayObject* pos, PyArrayObject* mass, int np, PyArrayObject* forcepos, int nf, double eps, double theta, int calc_force, int calc_potential, PyArrayObject* forcearray, PyArrayObject* potarray)
{
	struct gravoct_node *root;
	struct gravoct_particle *p;
	double min[3], max[3], boxsize, boxcenter[3],q;
	double thisforce[3],thispos[3],thispot;

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

	/* calculate forces and/or potentials */
	for(i=0; i<nf; i++) {
		for(j=0; j<3; j++) {
			thispos[j] = *(double *)PyArray_GETPTR2(forcepos,i,j);
		}
		gravoct_calc_accel(root, thispos, eps, theta, calc_force, calc_potential, thisforce, &thispot);
		/* save in forcearray/potarray */
		if(calc_force) {
            for(j=0; j<3; j++) {
                *(double *)PyArray_GETPTR2(forcearray,i,j) = thisforce[j];
            }
        }
        if(calc_potential) {
            *(double *)PyArray_GETPTR1(potarray,i) = thispot;
        }
	}

	/* destroy tree */
	gravoct_deltree(root);


	/* return None */
	return Py_None;
}

