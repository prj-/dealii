/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2009, 2010
 *         Timo Heister, University of Goettingen, 2009, 2010
 */


// @sect3{Include files}
//
// Most of the include files we need for this program have already been
// discussed in previous programs. In particular, all of the following should
// already be familiar friends:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/generic_linear_algebra.h>

// This program can use either PETSc or Trilinos for its parallel
// algebra needs. By default, if deal.II has been configured with
// PETSc, it will use PETSc. Otherwise, the following few lines will
// check that deal.II has been configured with Trilinos and take that.
//
// But there may be cases where you want to use Trilinos, even though
// deal.II has *also* been configured with PETSc, for example to
// compare the performance of these two libraries. To do this,
// add the following \#define to the source code:
// @code
// #define FORCE_USE_OF_TRILINOS
// @endcode
//
// Using this logic, the following lines will then import either the
// PETSc or Trilinos wrappers into the namespace `LA` (for "linear
// algebra). In the former case, we are also defining the macro
// `USE_PETSC_LA` so that we can detect if we are using PETSc (see
// solve() for an example where this is necessary).
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/derivative_approximation.h>

// The following, however, will be new or be used in new roles. Let's walk
// through them. The first of these will provide the tools of the
// Utilities::System namespace that we will use to query things like the
// number of processors associated with the current MPI universe, or the
// number within this universe the processor this job runs on is:
#include <deal.II/base/utilities.h>
// The next one provides a class, ConditionOStream that allows us to write
// code that would output things to a stream (such as <code>std::cout</code>
// on every processor but throws the text away on all but one of them. We
// could achieve the same by simply putting an <code>if</code> statement in
// front of each place where we may generate output, but this doesn't make the
// code any prettier. In addition, the condition whether this processor should
// or should not produce output to the screen is the same every time -- and
// consequently it should be simple enough to put it into the statements that
// generate output itself.
#include <deal.II/base/conditional_ostream.h>
// After these preliminaries, here is where it becomes more interesting. As
// mentioned in the @ref distributed module, one of the fundamental truths of
// solving problems on large numbers of processors is that there is no way for
// any processor to store everything (e.g. information about all cells in the
// mesh, all degrees of freedom, or the values of all elements of the solution
// vector). Rather, every processor will <i>own</i> a few of each of these
// and, if necessary, may <i>know</i> about a few more, for example the ones
// that are located on cells adjacent to the ones this processor owns
// itself. We typically call the latter <i>ghost cells</i>, <i>ghost nodes</i>
// or <i>ghost elements of a vector</i>. The point of this discussion here is
// that we need to have a way to indicate which elements a particular
// processor owns or need to know of. This is the realm of the IndexSet class:
// if there are a total of $N$ cells, degrees of freedom, or vector elements,
// associated with (non-negative) integral indices $[0,N)$, then both the set
// of elements the current processor owns as well as the (possibly larger) set
// of indices it needs to know about are subsets of the set $[0,N)$. IndexSet
// is a class that stores subsets of this set in an efficient format:
#include <deal.II/base/index_set.h>
// The next header file is necessary for a single function,
// SparsityTools::distribute_sparsity_pattern. The role of this function will
// be explained below.
#include <deal.II/lac/sparsity_tools.h>
// The final two, new header files provide the class
// parallel::distributed::Triangulation that provides meshes distributed
// across a potentially very large number of processors, while the second
// provides the namespace parallel::distributed::GridRefinement that offers
// functions that can adaptively refine such distributed meshes:
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>

namespace Step40
{
  using namespace dealii;

  template <int dim>
  class AdvectionField : public TensorFunction<1, dim>
  {
  public:
    virtual Tensor<1, dim> value(const Point<dim> &p) const override;

    // In previous examples, we have used assertions that throw exceptions in
    // several places. However, we have never seen how such exceptions are
    // declared. This can be done as follows:
    DeclException2(ExcDimensionMismatch,
                   unsigned int,
                   unsigned int,
                   << "The vector has size " << arg1 << " but should have "
                   << arg2 << " elements.");
    // The syntax may look a little strange, but is reasonable. The format is
    // basically as follows: use the name of one of the macros
    // <code>DeclExceptionN</code>, where <code>N</code> denotes the number of
    // additional parameters which the exception object shall take. In this
    // case, as we want to throw the exception when the sizes of two vectors
    // differ, we need two arguments, so we use
    // <code>DeclException2</code>. The first parameter then describes the
    // name of the exception, while the following declare the data types of
    // the parameters. The last argument is a sequence of output directives
    // that will be piped into the <code>std::cerr</code> object, thus the
    // strange format with the leading <code>@<@<</code> operator and the
    // like. Note that we can access the parameters which are passed to the
    // exception upon construction (i.e. within the <code>Assert</code> call)
    // by using the names <code>arg1</code> through <code>argN</code>, where
    // <code>N</code> is the number of arguments as defined by the use of the
    // respective macro <code>DeclExceptionN</code>.
    //
    // To learn how the preprocessor expands this macro into actual code,
    // please refer to the documentation of the exception classes. In brief,
    // this macro call declares and defines a class
    // <code>ExcDimensionMismatch</code> inheriting from ExceptionBase which
    // implements all necessary error output functions.
  };

  template <int dim>
  Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const
  {
    Tensor<1, dim> value;
    value[0] = 2;
    for (unsigned int i = 1; i < dim; ++i)
      value[i] = 1 + 0.8 * std::sin(8. * numbers::PI * p[0]);

    return value;
  }

  // Besides the advection field, we need two functions describing the source
  // terms (<code>right hand side</code>) and the boundary values. As
  // described in the introduction, the source is a constant function in the
  // vicinity of a source point, which we denote by the constant static
  // variable <code>center_point</code>. We set the values of this center
  // using the same template tricks as we have shown in the step-7 example
  // program. The rest is simple and has been shown previously.
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    static const Point<dim> center_point;
  };


  template <>
  const Point<1> RightHandSide<1>::center_point = Point<1>(-0.75);

  template <>
  const Point<2> RightHandSide<2>::center_point = Point<2>(-0.75, -0.75);

  template <>
  const Point<3> RightHandSide<3>::center_point = Point<3>(-0.75, -0.75, -0.75);

  // The only new thing here is that we check for the value of the
  // <code>component</code> parameter. As this is a scalar function, it is
  // obvious that it only makes sense if the desired component has the index
  // zero, so we assert that this is indeed the
  // case. <code>ExcIndexRange</code> is a global predefined exception
  // (probably the one most often used, we therefore made it global instead of
  // local to some class), that takes three parameters: the index that is
  // outside the allowed range, the first element of the valid range and the
  // one past the last (i.e. again the half-open interval so often used in the
  // C++ standard library):
  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    const double diameter = 0.1;
    return ((p - center_point).norm_square() < diameter * diameter ?
              0.1 / std::pow(diameter, dim) :
              0.0);
  }

  // Finally for the boundary values, which is just another class derived from
  // the <code>Function</code> base class:
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & p,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    const double sine_term = std::sin(16. * numbers::PI * p.norm_square());
    const double weight    = std::exp(5. * (1. - p.norm_square()));
    return weight * sine_term;
  }
  // @sect3{The <code>LaplaceProblem</code> class template}

  // Next let's declare the main class of this program. Its structure is
  // almost exactly that of the step-6 tutorial program. The only significant
  // differences are:
  // - The <code>mpi_communicator</code> variable that
  //   describes the set of processors we want this code to run on. In practice,
  //   this will be MPI_COMM_WORLD, i.e. all processors the batch scheduling
  //   system has assigned to this particular job.
  // - The presence of the <code>pcout</code> variable of type ConditionOStream.
  // - The obvious use of parallel::distributed::Triangulation instead of
  // Triangulation.
  // - The presence of two IndexSet objects that denote which sets of degrees of
  //   freedom (and associated elements of solution and right hand side vectors)
  //   we own on the current processor and which we need (as ghost elements) for
  //   the algorithms in this program to work.
  // - The fact that all matrices and vectors are now distributed. We use
  //   either the PETSc or Trilinos wrapper classes so that we can use one of
  //   the sophisticated preconditioners offered by Hypre (with PETSc) or ML
  //   (with Trilinos). Note that as part of this class, we store a solution
  //   vector that does not only contain the degrees of freedom the current
  //   processor owns, but also (as ghost elements) all those vector elements
  //   that correspond to "locally relevant" degrees of freedom (i.e. all
  //   those that live on locally owned cells or the layer of ghost cells that
  //   surround it).
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();

    void run(const PetscInt init, const PetscInt step);

  private:
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;
    const MappingQ1<dim>                      mapping;

    const FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    const QGauss<dim>     quadrature;
    const QGauss<dim - 1> quadrature_face;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };


  // @sect3{The <code>LaplaceProblem</code> class implementation}

  // @sect4{Constructor}

  // Constructors and destructors are rather trivial. In addition to what we
  // do in step-6, we set the set of processors we want to work on to all
  // machines available (MPI_COMM_WORLD); ask the triangulation to ensure that
  // the mesh remains smooth and free to refined islands, for example; and
  // initialize the <code>pcout</code> variable to only allow processor zero
  // to output anything. The final piece is to initialize a timer that we
  // use to determine how much compute time the different parts of the program
  // take:
  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , mapping()
    , fe(5)
    , dof_handler(triangulation)
    , quadrature(fe.tensor_degree() + 1)
    , quadrature_face(fe.tensor_degree() + 1)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
  {}



  // @sect4{LaplaceProblem::setup_system}

  // The following function is, arguably, the most interesting one in the
  // entire program since it goes to the heart of what distinguishes %parallel
  // step-40 from sequential step-6.
  //
  // At the top we do what we always do: tell the DoFHandler object to
  // distribute degrees of freedom. Since the triangulation we use here is
  // distributed, the DoFHandler object is smart enough to recognize that on
  // each processor it can only distribute degrees of freedom on cells it
  // owns; this is followed by an exchange step in which processors tell each
  // other about degrees of freedom on ghost cell. The result is a DoFHandler
  // that knows about the degrees of freedom on locally owned cells and ghost
  // cells (i.e. cells adjacent to locally owned cells) but nothing about
  // cells that are further away, consistent with the basic philosophy of
  // distributed computing that no processor can know everything.
  template <int dim>
  void LaplaceProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    // The next two lines extract some information we will need later on,
    // namely two index sets that provide information about which degrees of
    // freedom are owned by the current processor (this information will be
    // used to initialize solution and right hand side vectors, and the system
    // matrix, indicating which elements to store on the current processor and
    // which to expect to be stored somewhere else); and an index set that
    // indicates which degrees of freedom are locally relevant (i.e. live on
    // cells that the current processor owns or on the layer of ghost cells
    // around the locally owned cells; we need all of these degrees of
    // freedom, for example, to estimate the error on the local cells).
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Next, let us initialize the solution and right hand side vectors. As
    // mentioned above, the solution vector we seek does not only store
    // elements we own, but also ghost entries; on the other hand, the right
    // hand side vector only needs to have the entries the current processor
    // owns since all we will ever do is write into it, never read from it on
    // locally owned cells (of course the linear solvers will read from it,
    // but they do not care about the geometric location of degrees of
    // freedom).
    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    // The next step is to compute hanging node and boundary value
    // constraints, which we combine into a single object storing all
    // constraints.
    //
    // As with all other things in %parallel, the mantra must be that no
    // processor can store all information about the entire universe. As a
    // consequence, we need to tell the AffineConstraints object for which
    // degrees of freedom it can store constraints and for which it may not
    // expect any information to store. In our case, as explained in the
    // @ref distributed module, the degrees of freedom we need to care about on
    // each processor are the locally relevant ones, so we pass this to the
    // AffineConstraints::reinit function. As a side note, if you forget to
    // pass this argument, the AffineConstraints class will allocate an array
    // with length equal to the largest DoF index it has seen so far. For
    // processors with high MPI process number, this may be very large --
    // maybe on the order of billions. The program would then allocate more
    // memory than for likely all other operations combined for this single
    // array.
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    // The last part of this function deals with initializing the matrix with
    // accompanying sparsity pattern. As in previous tutorial programs, we use
    // the DynamicSparsityPattern as an intermediate with which we
    // then initialize the system matrix. To do so we have to tell the sparsity
    // pattern its size but as above there is no way the resulting object will
    // be able to store even a single pointer for each global degree of
    // freedom; the best we can hope for is that it stores information about
    // each locally relevant degree of freedom, i.e. all those that we may
    // ever touch in the process of assembling the matrix (the
    // @ref distributed_paper "distributed computing paper" has a long
    // discussion why one really needs the locally relevant, and not the small
    // set of locally active degrees of freedom in this context).
    //
    // So we tell the sparsity pattern its size and what DoFs to store
    // anything for and then ask DoFTools::make_sparsity_pattern to fill it
    // (this function ignores all cells that are not locally owned, mimicking
    // what we will do below in the assembly process). After this, we call a
    // function that exchanges entries in these sparsity pattern between
    // processors so that in the end each processor really knows about all the
    // entries that will exist in that part of the finite element matrix that
    // it will own. The final step is to initialize the matrix with the
    // sparsity pattern.
    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }



  // @sect4{LaplaceProblem::assemble_system}

  // The function that then assembles the linear system is comparatively
  // boring, being almost exactly what we've seen before. The points to watch
  // out for are:
  // - Assembly must only loop over locally owned cells. There
  //   are multiple ways to test that; for example, we could compare a cell's
  //   subdomain_id against information from the triangulation as in
  //   <code>cell->subdomain_id() ==
  //   triangulation.locally_owned_subdomain()</code>, or skip all cells for
  //   which the condition <code>cell->is_ghost() ||
  //   cell->is_artificial()</code> is true. The simplest way, however, is to
  //   simply ask the cell whether it is owned by the local processor.
  // - Copying local contributions into the global matrix must include
  //   distributing constraints and boundary values. In other words, we cannot
  //   (as we did in step-6) first copy every local contribution into the global
  //   matrix and only in a later step take care of hanging node constraints and
  //   boundary values. The reason is, as discussed in step-17, that the
  //   parallel vector classes do not provide access to arbitrary elements of
  //   the matrix once they have been assembled into it -- in parts because they
  //   may simply no longer reside on the current processor but have instead
  //   been shipped to a different machine.
  // - The way we compute the right hand side (given the
  //   formula stated in the introduction) may not be the most elegant but will
  //   do for a program whose focus lies somewhere entirely different.
  template <int dim>
  void LaplaceProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     update_values | update_quadrature_points |
                                       update_JxW_values | update_normal_vectors);

    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points =
      fe_face_values.get_quadrature().size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // ... obtain the values of right hand side and advection directions
    // at the quadrature points...
    AdvectionField<dim>         advection_field;
    RightHandSide<dim>          right_hand_side;
    std::vector<double>         rhs_values(n_q_points);
    std::vector<Tensor<1, dim>> advection_directions(fe_values.get_quadrature().size());
    std::vector<Tensor<1, dim>> face_advection_directions(fe_face_values.get_quadrature().size());
    BoundaryValues<dim>         boundary_values;
    std::vector<double>         face_boundary_values(fe_face_values.get_quadrature().size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const double delta = 0.1 * cell->diameter();
          cell_matrix = 0.;
          cell_rhs    = 0.;

          fe_values.reinit(cell);
          advection_field.value_list(
            fe_values.get_quadrature_points(),
            advection_directions);
          right_hand_side.value_list(
            fe_values.get_quadrature_points(), rhs_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) +=
                      ((fe_values.shape_value(i, q_point) +           // (phi_i +
                        delta * (advection_directions[q_point] *      // delta beta
                                 fe_values.shape_grad(i, q_point))) * // grad phi_i)
                       advection_directions[q_point] *                // beta
                       fe_values.shape_grad(j, q_point)) *            // grad phi_j
                      fe_values.JxW(q_point);                         // dx

                  cell_rhs(i) +=
                    (fe_values.shape_value(i, q_point) +           // (phi_i +
                     delta * (advection_directions[q_point] *      // delta beta
                              fe_values.shape_grad(i, q_point))) * // grad phi_i)
                    rhs_values[q_point] *                          // f
                    fe_values.JxW(q_point);                        // dx
                }
            }

          for (const auto &face : cell->face_iterators())
            if (face->at_boundary())
              {
                // Ok, this face of the present cell is on the boundary of the
                // domain. Just as for the usual FEValues object which we have
                // used in previous examples and also above, we have to
                // reinitialize the FEFaceValues object for the present face:
                fe_face_values.reinit(cell, face);

                // For the quadrature points at hand, we ask for the values of
                // the inflow function and for the direction of flow:
                boundary_values.value_list(
                  fe_face_values.get_quadrature_points(),
                  face_boundary_values);
                advection_field.value_list(
                  fe_face_values.get_quadrature_points(),
                  face_advection_directions);

                // Now loop over all quadrature points and see whether this face is on
                // the inflow or outflow part of the boundary. The normal
                // vector points out of the cell: since the face is at
                // the boundary, the normal vector points out of the domain,
                // so if the advection direction points into the domain, its
                // scalar product with the normal vector must be negative (to see why
                // this is true, consider the scalar product definition that uses a
                // cosine):
                for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                  if (fe_face_values.normal_vector(q_point) *
                        face_advection_directions[q_point] <
                      0.)
                    // If the face is part of the inflow boundary, then compute the
                    // contributions of this face to the global matrix and right
                    // hand side, using the values obtained from the
                    // FEFaceValues object and the formulae discussed in the
                    // introduction:
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          cell_matrix(i, j) -=
                            (face_advection_directions[q_point] *
                             fe_face_values.normal_vector(q_point) *
                             fe_face_values.shape_value(i, q_point) *
                             fe_face_values.shape_value(j, q_point) *
                             fe_face_values.JxW(q_point));

                        cell_rhs(i) -=
                          (face_advection_directions[q_point] *
                           fe_face_values.normal_vector(q_point) *
                           face_boundary_values[q_point] *
                           fe_face_values.shape_value(i, q_point) *
                           fe_face_values.JxW(q_point));
                      }
              }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

    // Notice that the assembling above is just a local operation. So, to
    // form the "global" linear system, a synchronization between all
    // processors is needed. This could be done by invoking the function
    // compress(). See @ref GlossCompress "Compressing distributed objects"
    // for more information on what is compress() designed to do.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  // @sect4{LaplaceProblem::solve}

  // Even though solving linear systems on potentially tens of thousands of
  // processors is by far not a trivial job, the function that does this is --
  // at least at the outside -- relatively simple. Most of the parts you've
  // seen before. There are really only two things worth mentioning:
  // - Solvers and preconditioners are built on the deal.II wrappers of PETSc
  //   and Trilinos functionality. It is relatively well known that the
  //   primary bottleneck of massively %parallel linear solvers is not
  //   actually the communication between processors, but the fact that it is
  //   difficult to produce preconditioners that scale well to large numbers
  //   of processors. Over the second half of the first decade of the 21st
  //   century, it has become clear that algebraic multigrid (AMG) methods
  //   turn out to be extremely efficient in this context, and we will use one
  //   of them -- either the BoomerAMG implementation of the Hypre package
  //   that can be interfaced to through PETSc, or a preconditioner provided
  //   by ML, which is part of Trilinos -- for the current program. The rest
  //   of the solver itself is boilerplate and has been shown before. Since
  //   the linear system is symmetric and positive definite, we can use the CG
  //   method as the outer solver.
  // - Ultimately, we want a vector that stores not only the elements
  //   of the solution for degrees of freedom the current processor owns, but
  //   also all other locally relevant degrees of freedom. On the other hand,
  //   the solver itself needs a vector that is uniquely split between
  //   processors, without any overlap. We therefore create a vector at the
  //   beginning of this function that has these properties, use it to solve the
  //   linear system, and only assign it to the vector we want at the very
  //   end. This last step ensures that all ghost elements are also copied as
  //   necessary.
  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector    completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

#ifdef USE_PETSC_LA
    LA::SolverGMRES solver(solver_control, mpi_communicator);
#else
    LA::SolverGMRES solver(solver_control);
#endif

    LA::MPI::PreconditionJacobi preconditioner;

    LA::MPI::PreconditionJacobi::AdditionalData data;

    preconditioner.initialize(system_matrix, data);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
  }



  // @sect4{LaplaceProblem::refine_grid}

  // The function that estimates the error and refines the grid is again
  // almost exactly like the one in step-6. The only difference is that the
  // function that flags cells to be refined is now in namespace
  // parallel::distributed::GridRefinement -- a namespace that has functions
  // that can communicate between all involved processors and determine global
  // thresholds to use in deciding which cells to refine and which to coarsen.
  //
  // Note that we didn't have to do anything special about the
  // KellyErrorEstimator class: we just give it a vector with as many elements
  // as the local triangulation has cells (locally owned cells, ghost cells,
  // and artificial ones), but it only fills those entries that correspond to
  // cells that are locally owned.
  template <int dim>
  void LaplaceProblem<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");
    Vector<float> gradient_indicator(triangulation.n_active_cells());
    DerivativeApproximation::approximate_gradient(dof_handler,
                                                  locally_relevant_solution,
                                                  gradient_indicator);
    // and they are cell-wise scaled by the factor $h^{1+d/2}$
    unsigned int cell_no = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      gradient_indicator(cell_no++) *=
        std::pow(cell->diameter(), 1 + 1.0 * dim / 2);
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, gradient_indicator, 0.3, 0.03);
    triangulation.execute_coarsening_and_refinement();
  }



  // @sect4{LaplaceProblem::output_results}

  // Compared to the corresponding function in step-6, the one here is a tad
  // more complicated. There are two reasons: the first one is that we do not
  // just want to output the solution but also for each cell which processor
  // owns it (i.e. which "subdomain" it is in). Secondly, as discussed at
  // length in step-17 and step-18, generating graphical data can be a
  // bottleneck in parallelizing. In step-18, we have moved this step out of
  // the actual computation but shifted it into a separate program that later
  // combined the output from various processors into a single file. But this
  // doesn't scale: if the number of processors is large, this may mean that
  // the step of combining data on a single processor later becomes the
  // longest running part of the program, or it may produce a file that's so
  // large that it can't be visualized any more. We here follow a more
  // sensible approach, namely creating individual files for each MPI process
  // and leaving it to the visualization program to make sense of that.
  //
  // To start, the top of the function looks like it usually does. In addition
  // to attaching the solution vector (the one that has entries for all locally
  // relevant, not only the locally owned, elements), we attach a data vector
  // that stores, for each cell, the subdomain the cell belongs to. This is
  // slightly tricky, because of course not every processor knows about every
  // cell. The vector we attach therefore has an entry for every cell that the
  // current processor has in its mesh (locally owned ones, ghost cells, and
  // artificial cells), but the DataOut class will ignore all entries that
  // correspond to cells that are not owned by the current processor. As a
  // consequence, it doesn't actually matter what values we write into these
  // vector entries: we simply fill the entire vector with the number of the
  // current MPI process (i.e. the subdomain_id of the current process); this
  // correctly sets the values we care for, i.e. the entries that correspond
  // to locally owned cells, while providing the wrong value for all other
  // elements -- but these are then ignored anyway.
  template <int dim>
  void LaplaceProblem<dim>::output_results(const unsigned int cycle) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "u", DataOut<dim>::type_dof_data);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "partitioning");
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    data_out.build_patches();

    // The next step is to write this data to disk. We write up to 8 VTU files
    // in parallel with the help of MPI-IO. Additionally a PVTU record is
    // generated, which groups the written VTU files.
    data_out.write_vtu_with_pvtu_record(
      "./", "solution-cg-" + std::to_string(dim) + "D", cycle, mpi_communicator, 2, 8);
  }



  // @sect4{LaplaceProblem::run}

  // The function that controls the overall behavior of the program is again
  // like the one in step-6. The minor difference are the use of
  // <code>pcout</code> instead of <code>std::cout</code> for output to the
  // console (see also step-17) and that we only generate graphical output if
  // at most 32 processors are involved. Without this limit, it would be just
  // too easy for people carelessly running this program without reading it
  // first to bring down the cluster interconnect and fill any file system
  // available :-)
  //
  // A functional difference to step-6 is the use of a square domain and that
  // we start with a slightly finer mesh (5 global refinement cycles) -- there
  // just isn't much of a point showing a massively %parallel program starting
  // on 4 cells (although admittedly the point is only slightly stronger
  // starting on 1024).
  template <int dim>
  void LaplaceProblem<dim>::run(const PetscInt init, const PetscInt n_cycles)
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    for (PetscInt cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, -1, 1);
            triangulation.refine_global(init);
          }
        else
          refine_grid();

        setup_system();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        assemble_system();
        solve();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(cycle);
          }

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
      }
  }
} // namespace Step40



// @sect4{main()}

// The final function, <code>main()</code>, again has the same structure as in
// all other programs, in particular step-6. Like the other programs that use
// MPI, we have to initialize and finalize MPI, which is done using the helper
// object Utilities::MPI::MPI_InitFinalize. The constructor of that class also
// initializes libraries that depend on MPI, such as p4est, PETSc, SLEPc, and
// Zoltan (though the last two are not used in this tutorial). The order here
// is important: we cannot use any of these libraries until they are
// initialized, so it does not make sense to do anything before creating an
// instance of Utilities::MPI::MPI_InitFinalize.
//
// After the solver finishes, the LaplaceProblem destructor will run followed
// by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize(). This order is
// also important: Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() calls
// <code>PetscFinalize</code> (and finalization functions for other
// libraries), which will delete any in-use PETSc objects. This must be done
// after we destruct the Laplace solver to avoid double deletion
// errors. Fortunately, due to the order of destructor call rules of C++, we
// do not need to worry about any of this: everything happens in the correct
// order (i.e., the reverse of the order of construction). The last function
// called by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() is
// <code>MPI_Finalize</code>: i.e., once this object is destructed the program
// should exit since MPI will no longer be available.
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step40;
      PetscBool flg;
      PetscInt  array[2],found = 2;
      PetscErrorCode ierr;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      ierr = PetscOptionsGetIntArray(NULL,NULL,"-two",array,&found,&flg);
      AssertThrow(ierr == 0 && (found == 2 || !flg), ExcPETScError(ierr));
      if (flg) {
          LaplaceProblem<2> laplace_problem;
          laplace_problem.run(array[0], array[1]);
      }
      found = 2;
      ierr = PetscOptionsGetIntArray(NULL,NULL,"-three",array,&found,&flg);
      AssertThrow(ierr == 0 && (found == 2 || !flg), ExcPETScError(ierr));
      if (flg) {
          LaplaceProblem<3> laplace_problem;
          laplace_problem.run(array[0], array[1]);
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
