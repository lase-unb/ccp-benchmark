
#include "reactions.h"

#include "rapidcsv.h"
#include "spark/collisions/basic_reactions.h"

using namespace spark::collisions;
using namespace spark::collisions::reactions;

namespace {
CrossSection load_cross_section(const std::filesystem::path& path, double energy_threshold) {
    CrossSection cs;
    const rapidcsv::Document doc(path.string(), rapidcsv::LabelParams(-1, -1),
                                 rapidcsv::SeparatorParams(';'));
    cs.energy = doc.GetColumn<double>(0);
    cs.cross_section = doc.GetColumn<double>(1);
    cs.threshold = energy_threshold;
    return cs;
}
}  // namespace

spark::collisions::Reactions<1, 3> ccp::reactions::load_electron_reactions(
    const std::filesystem::path& dir,
    const Parameters& par,
    spark::particle::ChargedSpecies<1, 3>& ions) {
    spark::collisions::Reactions<1, 3> electron_reactions;

    electron_reactions.push_back(std::make_unique<ElectronElasticCollision<1, 3>>(
        BasicCollisionConfig{par.m_he}, load_cross_section(dir / "Elastic_He.csv", 0.0)));

    electron_reactions.push_back(std::make_unique<ExcitationCollision<1, 3>>(
        BasicCollisionConfig{par.m_he}, load_cross_section(dir / "Excitation1_He.csv", 19.82)));

    electron_reactions.push_back(std::make_unique<ExcitationCollision<1, 3>>(
        BasicCollisionConfig{par.m_he}, load_cross_section(dir / "Excitation2_He.csv", 20.61)));

    electron_reactions.push_back(std::make_unique<IonizationCollision<1, 3>>(
        ions, par.tg, BasicCollisionConfig{par.m_he},
        load_cross_section(dir / "Ionization_He.csv", 24.59)));

    return electron_reactions;
}

spark::collisions::Reactions<1, 3> ccp::reactions::load_ion_reactions(
    const std::filesystem::path& dir,
    const Parameters& par) {
    spark::collisions::Reactions<1, 3> ion_reactions;
    ion_reactions.push_back(std::make_unique<IonElasticCollision<1, 3>>(
        BasicCollisionConfig{par.m_he}, load_cross_section(dir / "Isotropic_He.csv", 0.0)));

    ion_reactions.push_back(std::make_unique<ChargeExchangeCollision<1, 3>>(
        BasicCollisionConfig{par.m_he}, load_cross_section(dir / "Backscattering_He.csv", 0.0)));

    return ion_reactions;
}
