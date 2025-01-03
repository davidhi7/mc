use std::fmt::Debug;

pub fn bad_cmp_vec_unordered<T: Clone + Debug + Eq>(
    first: &Vec<T>,
    second: &Vec<T>,
) -> Result<(), String> {
    let first = first.clone();
    let mut second = second.clone();
    for element in first {
        let index = second
            .iter()
            .position(|e| *e == element)
            .ok_or(String::from(format!(
                "Element only occurs in `first`: {:?}",
                element
            )))?;
        second.swap_remove(index);
    }

    if second.len() == 0 {
        Ok(())
    } else {
        Err(format!(
            "Element only occurs in `second`: {:?}",
            second.into_iter().next().unwrap()
        ))
    }
}

#[test]
fn test_bad_cmp_vec_unordered() {
    assert!(bad_cmp_vec_unordered(&vec![0, 1, 2], &vec![0, 1, 2]).is_ok());
    assert!(bad_cmp_vec_unordered(&vec![0, 1, 2], &vec![2, 0, 1]).is_ok());
    assert!(bad_cmp_vec_unordered(&vec![0, 1, 2], &vec![0, 2]).is_err());
    assert!(bad_cmp_vec_unordered(&vec![0, 2], &vec![0, 1, 2]).is_err());
    assert!(bad_cmp_vec_unordered(&vec![], &vec![0, 1, 2]).is_err());
    assert!(bad_cmp_vec_unordered(&vec![0, 1, 2], &vec![]).is_err());
}
