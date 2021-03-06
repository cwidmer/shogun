/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Christian Widmer
 * Written (W) 2007-2010 Soeren Sonnenburg
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
 * Copyright (C) 2007-2013 Fraunhofer FIRST, MPG, TU-Berlin, MSKCC
 */

#ifndef _LIBLINEARMTL_H___
#define _LIBLINEARMTL_H___

#include <shogun/lib/config.h>


#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/optimization/liblinear/shogun_liblinear.h>
#include <shogun/lib/SGMatrixList.h>
#include <shogun/lib/SGNDArray.h>
#include <iostream>

namespace shogun
{

#ifdef HAVE_LAPACK


/** @brief class to implement LibLinear */
class CLibLinearMTL : public CLinearMachine
{
	public:
		/** default constructor  */
		CLibLinearMTL();


		/** constructor (using L2R_L1LOSS_SVC_DUAL as default)
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab training labels
		 */
		CLibLinearMTL(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab);

		/** destructor */
		virtual ~CLibLinearMTL();


		/** get classifier type
		 *
		 * @return the classifier type
		 */
		virtual EMachineType get_classifier_type() { return CT_LIBLINEAR; }

		/** set C
		 *
		 * @param c_neg C1
		 * @param c_pos C2
		 */
		inline void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }

		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline float64_t get_C2() { return C2; }

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/** set m_current_task
		 *
		 * @param ct new m_current_task
		 */
		inline void set_m_current_task(float64_t ct) 
		{ 
			m_current_task=ct;

			int32_t w_size = V[0].num_rows;

            //W = get_W();
            w.zero();

            for (int32_t m=0; m!=num_kernels; m++)
            {
                float64_t* tmp_w = W[m].get_column_vector(ct);
                for (int32_t i=0; i!=w_size; i++)
                {
                    w[i] += thetas[m] * tmp_w[i];
                }
            }

	        SG_INFO("current task updated, active w rewritten\n");
	        std::cout << "current task updated, active w rewritten" << std::endl;

		}

		/** get m_current_task
		 *
		 * @return m_current_task
		 */
		inline float64_t get_m_current_task() { return m_current_task; }

		/** set if bias shall be enabled
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** check if bias is enabled
		 *
		 * @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** @return object name */
		virtual const char* get_name() const { return "LibLinearMTL"; }

		/** get the maximum number of iterations liblinear is allowed to do */
		inline int32_t get_max_iterations()
		{
			return max_iterations;
		}

		/** set the maximum number of iterations liblinear is allowed to do */
		inline void set_max_iterations(int32_t max_iter=1000)
		{
			max_iterations=max_iter;
		}

		/** set number of kernels */
		inline void set_num_kernels(int32_t nk)
		{
			num_kernels = nk;
		}

		/** set number of tasks */
		inline void set_num_tasks(int32_t nt)
		{
			num_tasks = nt;
		}

		/** set optimize_theta */
		inline void set_optimize_theta(float64_t ot)
		{
			optimize_theta = ot;
		}

		/** get optimize_theta */
		inline float64_t get_optimize_theta()
		{
			return optimize_theta;
		}

		/** set p_norm */
		inline void set_p_norm(float64_t pn)
		{
			p_norm = pn;
		}

		/** get p_norm */
		inline float64_t get_p_norm()
		{
			return p_norm;
		}

		/** set thetas */
		inline void set_thetas(SGVector<float64_t> t)
		{
			thetas = t;
		}

		/** get thetas */
		inline SGVector<float64_t> get_thetas()
		{
			return thetas;
		}

		/** set the linear term for qp */
		inline void set_linear_term(SGVector<float64_t> linear_term)
		{
			if (!m_labels)
				SG_ERROR("Please assign labels first!\n")

			int32_t num_labels=m_labels->get_num_labels();

			if (num_labels!=linear_term.vlen)
			{
				SG_ERROR("Number of labels (%d) does not match number"
						" of entries (%d) in linear term \n", num_labels,
						linear_term.vlen);
			}

			m_linear_term = linear_term;
		}

		/** set task indicator for lhs */
		inline void set_task_indicator_lhs(SGVector<int32_t> ti)
		{
			task_indicator_lhs = ti;
		}

		/** set task indicator for rhs */
		inline void set_task_indicator_rhs(SGVector<int32_t> ti)
		{
			task_indicator_rhs = ti;
		}

		/** set Q */
		inline void setup_Q(int32_t nk, int32_t nt)
		{
			num_kernels = nk;
			num_tasks = nt;
			Q = SGMatrixList<float64_t>(num_kernels, num_tasks, num_tasks);
			Q_inv = SGMatrixList<float64_t>(num_kernels, num_tasks, num_tasks);
		}

		/** get Q[idx] */
		inline SGMatrix<float64_t> get_Qi(int32_t idx)
		{
			return Q[idx];
		}

		/** set Q[idx] */
		inline void set_Qi(SGMatrix<float64_t> qm, int32_t idx)
		{
			Q[idx] = qm;
		}

		/** get Q_inv[idx] */
		inline SGMatrix<float64_t> get_Q_inv_i(int32_t idx)
		{
			return Q_inv[idx];
		}

		/** set Q_inv[idx] */
		inline void set_Q_inv_i(SGMatrix<float64_t> qm, int32_t idx)
		{
			Q_inv[idx] = qm;
		}

		/** get V
		 *
		 * @return matrix of weight vectors
		 */
		inline SGMatrixList<float64_t> get_V()
		{
			return V;
		}

		/** get W
		 *
		 * @return matrix of weight vectors
		 */
		inline SGMatrixList<float64_t> get_W()
		{
			
			int32_t w_size = V[0].num_rows;

			SGMatrixList<float64_t> new_W = SGMatrixList<float64_t>(num_kernels, w_size, num_tasks);
			for (int32_t m=0; m<num_kernels; m++)
			{
				for (int32_t s=0; s<num_tasks; s++)
				{
					float64_t* v_s = V[m].get_column_vector(s);
					for (int32_t t=0; t<num_tasks; t++)
					{
						float64_t sim_st = thetas[m] * Q_inv[m](s,t); //TODO check if Q_inv is correct
						for(int32_t i=0; i<w_size; i++)
						{
							//new_W[m].matrix[t*w_size + i] += sim_ts * v_s[i];
							new_W[m](i,t) += sim_st * v_s[i];
						}
					}
				}
			}

			return new_W;
		}

		/** get V_m
		 *
		 * @return matrix of working vectors for kernel m
		 */
		inline SGMatrix<float64_t> get_Vm(int32_t m)
		{
			return V[m];
		}

		/** get W_m
		 *
		 * @return matrix of weight vector for kernel m
		 */
		inline SGMatrix<float64_t> get_Wm(int32_t m)
		{
			
			int32_t w_size = V[0].num_rows;

			SGMatrix<float64_t> Wm = SGMatrix<float64_t>(w_size, num_tasks);
			Wm.zero();

			for (int32_t s=0; s<num_tasks; s++)
			{
				float64_t* v_s = V[m].get_column_vector(s);
				for (int32_t t=0; t<num_tasks; t++)
				{
					float64_t sim_st = thetas[m] * Q_inv[m](s,t);
					for(int32_t i=0; i<w_size; i++)
					{
						Wm(i,t) += sim_st * v_s[i];
					}
				}
			}

			return Wm;
		}

		/** get alphas
		 *
		 * @return matrix of example weights alphas
		 */
		inline SGVector<float64_t> get_alphas()
		{
			return alphas;
		}

		/** compute primal objective
		 *
		 * @return primal objective
		 */
		virtual float64_t compute_primal_obj();

		/** compute dual objective
		 *
		 * @return dual objective
		 */
		virtual float64_t compute_dual_obj();

		/** compute dual objective
		 *
		 * @return dual objective
		 */
		virtual float64_t compute_dual_obj_alphas();

		/** compute duality gap
		 *
		 * @return duality gap
		 */
		virtual float64_t compute_duality_gap();


	protected:
		/** train linear SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	private:
		/** set up parameters */
		void init();

		void solve_l2r_l1l2_svc(
			const problem *prob, double eps, double Cp, double Cn);


	protected:
		/** C1 */
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** if bias shall be used */
		bool use_bias;
		/** epsilon */
		float64_t epsilon;
		/** maximum number of iterations */
		int32_t max_iterations;

		/** precomputed linear term */
		SGVector<float64_t> m_linear_term;

		/** keep track of alphas */
		SGVector<float64_t> alphas;

		/** vector of MKL variables */
		SGVector<float64_t> thetas;

		/** p-norm for MKL */
		float64_t p_norm;

		/** set number of tasks */
		int32_t num_tasks;

		/** set number of kernels */
		int32_t num_kernels;

		/** task indicator left hand side */
		SGVector<int32_t> task_indicator_lhs;

		/** task indicator right hand side */
		SGVector<int32_t> task_indicator_rhs;

		/** multi Q */
		SGMatrixList<float64_t> Q;

		/** multi Q^-1 */
		SGMatrixList<float64_t> Q_inv;

		/** parameter matrix n * d */
		SGMatrixList<float64_t> V;

		/** parameter matrix n * d */
		SGMatrixList<float64_t> W;

		/** duality gap */
		float64_t duality_gap;

		/** active w (for prediction) **/
		int32_t m_current_task;

		/** flag indicating if MKL is turned on **/
		bool optimize_theta;

};

#endif //HAVE_LAPACK

} /* namespace shogun  */

#endif //_LIBLINEARMTL_H___
