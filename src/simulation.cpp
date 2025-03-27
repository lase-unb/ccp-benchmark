#include "simulation.h"

#include <spark/collisions/mcc.h>
#include <spark/constants/constants.h>
#include <spark/em/electric_field.h>
#include <spark/em/poisson.h>
#include <spark/interpolate/field.h>
#include <spark/interpolate/weight.h>
#include <spark/particle/boundary.h>
#include <spark/particle/pusher.h>
#include <spark/random/random.h>
#include <spark/spatial/grid.h>

#include "reactions.h"

namespace {
auto maxwellian_emitter(double t, double l, double m) {
    return [l, t, m](spark::core::Vec<3>& v, spark::core::Vec<1>& x) {
        x.x = l * spark::random::uniform();
        double vth = std::sqrt(spark::constants::kb * t / m);
        v = {spark::random::normal(0.0, vth), spark::random::normal(0.0, vth),
             spark::random::normal(0.0, vth)};
    };
}
}  // namespace

namespace ccp {
Simulation::Simulation(const Parameters& parameters, const std::string& data_path)
    : parameters_(parameters), data_path_(data_path), state_(StateInterface(*this)) {}

void Simulation::run() {
    set_initial_conditions();

    auto electron_collisions = load_electron_collisions();
    auto ion_collisions = load_ion_collisions();
    spark::core::TMatrix<spark::core::Vec<1>, 1> force_electrons_, force_ions_;

    auto poisson_solver = spark::em::ThomasPoissonSolver1D(parameters_.nx, parameters_.dx);

    events().notify(Event::Start, state_);

    for (step = 0; step < parameters_.n_steps; ++step) {
        spark::interpolate::weight_to_grid(electrons_, electron_density_);
        spark::interpolate::weight_to_grid(ions_, ion_density_);
        spark::em::charge_density(parameters_.particle_weight, ion_density_, electron_density_,
                                  rho_field_);

        const double boundary_voltage =
            parameters_.volt * std::sin(2.0 * spark::constants::pi * parameters_.f *
                                        parameters_.dt * static_cast<double>(step));
        poisson_solver.solve(rho_field_.data().data(), phi_field_.data().data(), 0.0,
                             boundary_voltage);

        spark::em::electric_field(phi_field_, electric_field_.data());

        spark::interpolate::field_at_particles(electric_field_, electrons_, force_electrons_);
        spark::interpolate::field_at_particles(electric_field_, ions_, force_ions_);

        spark::particle::move_particles(electrons_, force_electrons_, parameters_.dt);
        spark::particle::move_particles(ions_, force_ions_, parameters_.dt);

        spark::particle::apply_absorbing_boundary(electrons_, 0, parameters_.l);
        spark::particle::apply_absorbing_boundary(ions_, 0, parameters_.l);

        electron_collisions.react_all();
        ion_collisions.react_all();

        events().notify(Event::Step, state_);
    }

    events().notify(Event::End, state_);
}

Events<Simulation::Event, Simulation::EventAction>& Simulation::events() {
    return events_;
}

void Simulation::set_initial_conditions() {
    // Charged species
    electrons_ = spark::particle::ChargedSpecies<1, 3>(-spark::constants::e, spark::constants::m_e);
    electrons_.add(parameters_.n_initial,
                   maxwellian_emitter(parameters_.te, parameters_.l, spark::constants::m_e));

    ions_ = spark::particle::ChargedSpecies<1, 3>(spark::constants::e, parameters_.m_he);
    ions_.add(parameters_.n_initial,
              maxwellian_emitter(parameters_.ti, parameters_.l, parameters_.m_he));

    // Fields
    electron_density_ = spark::spatial::UniformGrid<1>({parameters_.l}, {parameters_.nx});
    ion_density_ = spark::spatial::UniformGrid<1>({parameters_.l}, {parameters_.nx});
    rho_field_ = spark::spatial::UniformGrid<1>({parameters_.l}, {parameters_.nx});
    phi_field_ = spark::spatial::UniformGrid<1>({parameters_.l}, {parameters_.nx});
    electric_field_ =
        spark::spatial::TUniformGrid<spark::core::Vec<1>, 1>({parameters_.l}, {parameters_.nx});
}

spark::collisions::MCCReactionSet<1, 3> Simulation::load_electron_collisions() {
    // Load electron reactions
    auto electron_reactions = reactions::load_electron_reactions(data_path_, parameters_, ions_);

    spark::collisions::ReactionConfig<1, 3> electron_reaction_config{
        parameters_.dt, parameters_.dx,
        std::make_unique<spark::collisions::StaticUniformTarget<1, 3>>(parameters_.ng,
                                                                       parameters_.tg),
        std::move(electron_reactions), spark::collisions::RelativeDynamics::FastProjectile};

    return {electrons_, std::move(electron_reaction_config)};
}

spark::collisions::MCCReactionSet<1, 3> Simulation::load_ion_collisions() {
    // Load ion reactions
    auto ion_reactions = reactions::load_ion_reactions(data_path_, parameters_);

    spark::collisions::ReactionConfig<1, 3> ion_reaction_config{
        parameters_.dt, parameters_.dx,
        std::make_unique<spark::collisions::StaticUniformTarget<1, 3>>(parameters_.ng,
                                                                       parameters_.tg),
        std::move(ion_reactions), spark::collisions::RelativeDynamics::SlowProjectile};

    return {ions_, std::move(ion_reaction_config)};
}
}  // namespace ccp
