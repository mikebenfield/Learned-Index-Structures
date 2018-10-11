//! Run the Python script `train.py` to train a hierarchy of models.

use std::ffi::OsStr;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use tempfile::NamedTempFile;

pub fn train<P>(data: &[f32], layers: usize, width: usize, threshold: usize, py_path: &P)
where
    P: AsRef<Path>,
{
    train0(data, layers, width, threshold, py_path.as_ref());
}

fn train0(data: &[f32], layers: usize, width: usize, threshold: usize, py_path: &Path) {
    let os: &OsStr = py_path.as_ref();
    let file_name = NamedTempFile::new().expect("Unable to create temp file");
    {
        let mut file = File::create(&file_name).expect("Unable to open temp file");
        for &datum in data.iter() {
            writeln!(file, "{}", datum).expect("Unable to write to temp file");
        }
    }
    Command::new("python3.6")
        .arg(os)
        .args(&[
            "--layers",
            &format!("{}", layers),
            "--width",
            &format!("{}", width),
            "--threshold",
            &format!("{}", threshold),
        ])
        .arg("--index")
        .arg(file_name.path())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .output()
        .expect("Failed to execute Python script");
}
