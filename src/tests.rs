use std::fmt::Debug;

pub fn cmp_vec_unordered<T: Clone + Debug + Eq>(left: &Vec<T>, right: &Vec<T>) -> Result<(), ()> {
    let first: Vec<T> = left.clone();
    let mut second = right.clone();
    for element in first {
        let index = second.iter().position(|e| *e == element).ok_or_else(|| {
            eprintln!(
                "assertion `left == right` failed\n  left: {:?}\n right: {:?}",
                left, right
            );
        })?;
        second.swap_remove(index);
    }

    if second.len() == 0 {
        Ok(())
    } else {
        eprintln!(
            "assertion `left == right` failed\n  left: {:?}\n right: {:?}",
            left, right
        );
        Err(())
    }
}

#[test]
fn test_bad_cmp_vec_unordered() {
    assert!(cmp_vec_unordered(&vec![0, 1, 2], &vec![0, 1, 2]).is_ok());
    assert!(cmp_vec_unordered(&vec![0, 1, 2], &vec![2, 0, 1]).is_ok());
    assert!(cmp_vec_unordered(&vec![0, 1, 2], &vec![0, 2]).is_err());
    assert!(cmp_vec_unordered(&vec![0, 2], &vec![0, 1, 2]).is_err());
    assert!(cmp_vec_unordered(&vec![], &vec![0, 1, 2]).is_err());
    assert!(cmp_vec_unordered(&vec![0, 1, 2], &vec![]).is_err());
}
