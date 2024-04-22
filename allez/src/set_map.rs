
extern crate ndarray_linalg;

use ndarray_linalg::{cholesky::*, Inverse, Trace};
use ndarray_csv::Array2Reader;
use rand::{
    Rng,
    seq::IteratorRandom,
};
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Dim, Ix, ViewRepr};
use ndarray_stats::CorrelationExt;
use statrs::distribution::{FisherSnedecor, ContinuousCDF, ChiSquared};

#[derive(Debug)]
pub struct Whole<G, S> {
    id: S,
    parts: Vec<Part<G>>,
}

pub struct EmptyWhole {}

impl<G, S> Whole<G, S> 
where S: std::fmt::Debug + std::cmp::PartialEq<str>, G: std::fmt::Debug {
    pub fn mean(&self) -> Result<Array1<f64>, EmptyWhole> {
        match self.parts.len() {
            0 => Err(EmptyWhole {}),
            _ => {
                let nlists = self.parts[0].score.len();
                let nparts = self.parts.len() as f64;
                let mu = self.parts.iter()
                    .fold(Array1::<f64>::zeros(nlists), |curr, x| curr + &x.score) / nparts;
        
                Ok(mu)
            }
        }
    }
    pub fn sum(&self) -> Result<Array1<f64>, EmptyWhole> {
        match self.parts.len() {
            0 => Err(EmptyWhole {}),
            _ => {
                let nlists = self.parts[0].score.len();
                let nparts = self.parts.len() as f64;
                let mu = self.parts.iter()
                    .fold(Array1::<f64>::zeros(nlists), |curr, x| curr + &x.score);
        
                Ok(mu)
            }
        }
    }
    fn variance_scalar(&self, nparts_global: f64) -> f64 {
        let m = self.nparts() as f64;
        (nparts_global * m)  / (nparts_global - m)
    }
    
    pub fn calculate_z(&self, mu: &Array1<f64>, sigma_cholesky_inv: &Array2<f64>, nparts_global: usize) -> Array1<f64> {
        match self.mean() {
            Ok(x) => (self.variance_scalar(nparts_global as f64).sqrt() * sigma_cholesky_inv).dot( &(x- mu) ),
            Err(x) => Array1::<f64>::zeros(mu.len())
        }
    }
    pub fn calculate_t_squared(&self, mu: &Array1<f64>, sigma_inv: &Array2<f64>, nparts_global: usize) -> f64 {
        match self.mean() {
            Ok(x) => {

                let sigma = self.variance_scalar(nparts_global as f64) * sigma_inv;
                let centered_x = x - mu;
        
                centered_x.t().dot(&sigma).dot(&centered_x)
            },
            _ => 0.0
        }
    }

    fn calc_sigma(&self) -> Result<Array2<f64>, EmptyWhole> {
        let nlists = self.nlists();
        let mut scores = Array2::<f64>::zeros((self.nparts(), self.nlists()));
        self.parts.iter()
            .zip(scores.outer_iter_mut())
            .for_each(|(part, mut score)| {
                score.assign(&(&score + part.score.clone()));
                // let score = &Array::from_shape_vec((1, nlists as usize), part.score.to_vec()).unwrap();
                // sigma = &sigma + score.t().dot(score);
            });
        Ok(scores.t().cov(1.0).unwrap())
    }

    pub fn calculate_wishart_m(&self, sigma_global_inv: &Array2<f64>, nparts_global: f64) -> Result<f64, EmptyWhole> {
        let nlists = self.nlists() as f64;

        if self.nparts() <= 1.0 as usize {
            return Err(EmptyWhole {})
        }

        let m;
        match self.calc_sigma() {
            Ok(sigma) => {
                let sigma_complement = (sigma_global_inv - self.nparts() as f64 * sigma.clone()) / (nparts_global - self.nparts() as f64);
                let sigmas = [sigma, sigma_complement];
                let ns = [self.nparts() as f64, nparts_global - self.nparts() as f64];

                m = ( nparts_global / 2.0 ) * (
                    sigmas.iter()
                        .zip(ns.iter())
                        .map(|( sigma, n )| (n / nparts_global) * 
                            sigma.dot(sigma_global_inv).dot(sigma).dot(sigma_global_inv).trace().unwrap()
                        )
                        .sum::<f64>() - 
                    sigmas.iter()
                        .zip(ns.iter())
                        .map(|(sigma1, n1)| {
                            sigmas.iter()
                            .zip(ns.iter())
                                .map(|(sigma2, n2)| {
                                    (n1 * n2 / nparts_global.powf(2.0)) * 
                                        sigma1.dot(sigma_global_inv).dot(sigma2).dot(sigma_global_inv).trace().unwrap()
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                );
            },
            _ => {m = 0.0;}
        }


        Ok(m)
    }


    pub fn parts(&self) -> &Vec<Part<G>> {&self.parts}
    pub fn id(&self) -> &S {&self.id}
    pub fn nparts(&self) -> usize {self.parts.len()}
    pub fn nlists(&self) -> usize {self.parts[0].score.len()}
}

#[derive(Clone, Debug)]
pub struct Part<G> {
    id: G,
    score: Array1<f64>,
}

pub struct PartWholeMap<G, S> {
    parts: Vec<Part<G>>, // Vector of parts, such as genes
    wholes: Vec<Whole<G, S>>, // Vector of wholes, such as gene sets
    mu: Array1<f64>,
    sigma: Array2<f64>,
}

impl<G, S> PartWholeMap<G, S> 
where G: Clone + std::fmt::Debug, S: Clone + std::fmt::Debug + std::cmp::PartialEq<str>{
    pub fn nparts(&self) -> usize {self.parts.len()}
    pub fn nlists(&self) -> usize {self.parts[0].score.len()}
    pub fn nwholes(&self) -> usize {self.wholes.len()}
    pub fn mean(&self) -> &Array1<f64> {&self.mu}
    pub fn sigma(&self) -> &Array2<f64> {&self.sigma}
    pub fn cholesky(&self) -> Array2<f64> {
        self.sigma.cholesky(UPLO::Lower).unwrap()
    }
    pub fn wholes(&self) -> &Vec<Whole<G, S>> {&self.wholes}
    pub fn scores(&self) -> Array2<f64> {
        let mut scores = Array2::<f64>::zeros((self.nparts(), self.nlists()));
        scores.outer_iter_mut()
            .zip(self.parts.iter())
            .for_each(|(mut scores, part)| {
                scores.assign(&part.score);
            });

        scores
    }

    pub fn new(score_name: &str, map_name: &str, part_ids: Vec<G>, whole_ids: Vec<S>) -> Self {

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(score_name)
            .expect("Score file not found");
        let mut scores: Array2<f64> = rdr.deserialize_array2_dynamic().expect("Score deserialization error");
        assert!(scores.dim().0 == part_ids.len());

        let nlists = scores.dim().1;

        let mut partwholemap = PartWholeMap::<G, S> {
            wholes: whole_ids.iter().map(|id| Whole {id: id.clone(), parts: vec![]}).collect(),
            parts: scores.outer_iter().zip(part_ids.iter()).map(|(score, id)| Part {id: id.clone(), score: score.to_owned()}).collect(),
            mu: Array1::<f64>::zeros(nlists),
            sigma: Array2::<f64>::eye(nlists),
        };

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(map_name)
            .expect("Map file not found");
        for result in rdr.deserialize() {
            let (whole, part): (usize, usize) = result.expect("Deserialization error");
            let part = Part{id: part_ids[part].clone(), score: scores.row(part).to_owned()};
            partwholemap.wholes[whole].parts.push(part);
        }



        // map.iter()
        //     .for_each(|(whole, part)| {
        //         partwholemap.wholes[*whole].parts.push(Part{id: part_ids[*part], score: scores.row(1).to_owned()});
        //         // indicators.iter()
        //         //     .zip(partwholemap.parts.iter())
        //         //     .for_each(|(&indicator, part)| {
        //         //         if indicator {
        //         //             whole.parts.push(part.clone());
        //         //         }
        //         //     });
        //     });

        partwholemap.calc_mean();
        partwholemap.calc_sigma();

        partwholemap
    }

    fn calc_mean(&mut self) {
        let nparts = self.parts.len();
        let nlists = self.parts[0].score.len();
        assert!(self.parts.iter().all(|x| x.score.len() == nlists));
        self.mu = self.parts.iter().fold(Array1::<f64>::zeros(nlists), |curr, x| curr + &x.score) / nparts as f64
    }

    fn calc_sigma(&mut self) {
        let mut centered_scores = Array2::<f64>::zeros((self.nparts(), self.nlists()));
        let mu = self.mu.clone();
        centered_scores.outer_iter_mut()
            .zip(self.parts.iter())
            .for_each(|(mut score, part)| {
                score.assign(&(&score + part.score.clone() - mu.clone()));
            });
        self.sigma = centered_scores.t().cov(1.0).unwrap();
    }

    fn variance_scalar(&self, m: f64) -> f64 {
        let g = self.nparts() as f64;
        (g * m)  / (g - m)
    }

    pub fn calculate_z(&self) -> Array2<f64> {
        let mut z_scores = Array2::<f64>::zeros((self.nwholes(), self.nlists()));
        let sigma = &self.cholesky().inv().unwrap();
        let mu = self.mean();

        z_scores.outer_iter_mut()
            .zip(self.wholes.iter())
            .for_each(|(mut scores, whole)| {
                // println!("whole mean: {:?}", whole.mean());
                // println!("mu: {:?}", mu);
                // println!("sd: {:?}", scalar.sqrt() * &std);
                // println!("whole length: {:?}", whole.parts.len());
                // println!("whole: {:?}", whole);
                // panic!();
                scores.assign( &( whole.calculate_z(mu, sigma, self.nparts()) ));
            });

        z_scores
    }

    pub fn calculate_t_squared(&self) -> Array1<f64> {
        let mut t_squared = Array1::<f64>::zeros(self.nwholes());
        let sigma = &self.sigma().inv().unwrap();
        let mu = self.mean();

        t_squared.iter_mut()
            .zip(self.wholes.iter())
            .for_each(|(score, whole)| {
                // println!("whole mean: {:?}", whole.mean());
                // println!("mu: {:?}", mu);
                // println!("sd: {:?}", scalar.sqrt() * &std);
                // println!("whole length: {:?}", whole.parts.len());
                // println!("whole: {:?}", whole);
                // panic!();
                *score = whole.calculate_t_squared(mu, sigma, self.nparts()) ;
            });

        t_squared
    }

    pub fn calculate_t_squared_pvalue(&self, tscores: Array1<f64>) -> Array1<Option<f64>> {
        let nglobal_parts = self.nparts() as f64;
        let nlists = self.nlists() as f64;
        self.wholes.iter()
            .zip(tscores.iter())
            .map(|(whole, tscore)| {
                let nlocal_parts = whole.nparts() as f64;
                if nlocal_parts <= nlists {
                    return None
                }
                let t = (nlocal_parts - nlists) / (nlists * (nlocal_parts - 1.0) );
                let score = t * tscore;
                let fdist = FisherSnedecor::new(nlists, nlocal_parts - nlists).unwrap();
                Some(fdist.sf(score))
            })
            .collect()
    }

    pub fn calculate_wishart_m(&mut self) -> Array1<f64> {
        let mut m_scores = Array1::<f64>::zeros(self.nwholes());
        let nglobal_parts = self.nparts() as f64;
        let nlists = self.nlists();
        // let mut sigma = Array2::<f64>::zeros((nlists, nlists));

        // self.scores().outer_iter()
        //     .for_each(|score| {
        //         let score = &Array::from_shape_vec((1, nlists as usize), score.to_vec()).unwrap();
        //         println!("{:?}", score.t().dot(score));
        //         sigma = &sigma + score.t().dot(score);
        // });

        // println!("{:?}", self.scores().row(0).dot(&self.scores().row(0)));
        // println!("{:?}", Array::from_shape_vec((1, nlists as usize), self.scores().row(0).to_vec()).unwrap());
        // println!("{:?}", sigma);

        let sigma_global_inv = &self.sigma.inv().unwrap();
        
        self.wholes.iter_mut()
            .zip(m_scores.iter_mut())
            .for_each(|(whole, m_score)| {
                match whole.calculate_wishart_m(sigma_global_inv, nglobal_parts) {
                    Ok(m) => {*m_score = m;}
                    _ => {}
                }
            });

        m_scores

    }


    pub fn calculate_wishart_p_value(&self, m_scores: &Array1<f64>) -> Array1<f64> {
        let nlists = self.nlists() as f64;
        let chisq = ChiSquared::new(nlists * (nlists + 1.0) / 2.0).unwrap();
        m_scores.iter()
            .map(|&m_score| {
                chisq.sf(m_score) 
            })
            .collect::<Array1<f64>>()
    }
}