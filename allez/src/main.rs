pub mod set_map;


use ndarray::arr2;
use ndarray_csv::Array2Reader;

fn main() {
    // let scores = &arr2(&[
    //     [1.0],
    //     [1.0],
    //     [0.0],
    //     [1.0],
    //     [0.0],
    //     [0.0],
    //     [0.0],
    // ]);

    // let map = &arr2(&[
    //     [true, false, false, true, false, true, false],
    //     [false, false, true, false, true, true, false],
    // ]);

    let score_name = "/mnt/hdd/Allez/allez/fludata.csv";
    let map_name = "/mnt/hdd/Allez/allez/flumap.csv";

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("/mnt/hdd/Allez/allez/flugo.csv")
        .expect("Whole names file not found");
    let mut go_ids: Vec<String> = vec![];
    for record in  rdr.records().into_iter() {
        go_ids.push(record.unwrap()[0].to_string());
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("/mnt/hdd/Allez/allez/flugenes.csv")
        .expect("Whole names file not found");
    let mut symbol_ids: Vec<String> = vec![];
    for record in  rdr.records().into_iter() {
        symbol_ids.push(record.unwrap()[0].to_string());
    }

    let mut allez = set_map::PartWholeMap::new(score_name, map_name, symbol_ids, go_ids.clone());

    println!("nparts: {:?}", allez.nparts());
    println!("Mean is: {:?}", allez.mean());
    println!("Sigma is: {:?}", allez.sigma());
    println!("Cholesky is: {:?}", allez.cholesky());
    println!("Z-scores are:");
    let mut wtr = csv::Writer::from_path("/mnt/hdd/Allez/allez/fluresults.csv").unwrap();
    wtr.write_record(&["go", "z1", "z2", "z3", "z4", "t", "p", "nparts", "sum_scores_1", "sum_scores_2", "sum_scores_3", "sum_scores_4", "m_score", "wishart_p"]).unwrap();

    let ts = allez.calculate_t_squared();

    let m_scores = allez.calculate_wishart_m();
    let m_pvals = allez.calculate_wishart_p_value(&m_scores);

    allez.calculate_z().outer_iter()
        .zip(allez.calculate_t_squared().iter())
        .zip(allez.calculate_t_squared_pvalue(ts).iter())
        .zip(allez.wholes().iter())
        .zip(m_scores.iter())
        .zip(m_pvals.iter())
        .for_each(|(((((z, t), p), whole), m), p_m)| {
            println!("{:?}", whole.id());
            println!("\tz = {:?}", z[0]);
            println!("\tt2 = {:?}", t);
            println!("\tp = {:?}", p);
            let p = match p {
                Some(p) => (p*5515.0).to_string(),
                _ => "".to_string(),
            };

            let sum_scores = match whole.sum() {
                Ok(x) => vec![
                    x[0].to_string(),
                    x[1].to_string(),
                    x[2].to_string(),
                    x[3].to_string(),
                    ],
                _ => vec![
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                ]
            };
            
            wtr.write_record(&[whole.id(), 
                &z[0].to_string(), 
                &z[1].to_string(), 
                &z[2].to_string(), 
                &z[3].to_string(), 
                &t.to_string(), 
                &p.to_string(), 
                &whole.parts().len().to_string(), 
                &sum_scores[0],
                &sum_scores[1],
                &sum_scores[2],
                &sum_scores[3],
                &m.to_string(),
                &(p_m * 5515.0).to_string(),
            ]).unwrap();
        });
    wtr.flush().unwrap();
        // .for_each(|(z)| println!("GO: z = {:?}", z));

    
}
