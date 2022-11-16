use std::vec::Vec;
use std::time::Instant;
use std::thread;
use std::time;
use std::io::{self, Write, BufWriter};
use std::fs::File;

use byteorder::{ WriteBytesExt, LittleEndian};

use soapysdr;
use soapysdr::Args;
use soapysdr::Direction;
use soapysdr::TxStream;
use soapysdr::RxStream;

use num::complex::Complex;
use num::traits::FloatConst;

use num_traits::cast::FromPrimitive;


fn write_cfile<W: Write>(src_buf: &[Complex<f32>], mut dest_file: W) -> io::Result<()> {
    for sample in src_buf {
        dest_file.write_f32::<LittleEndian>(sample.re)?;
        dest_file.write_f32::<LittleEndian>(sample.im)?;
    }
    Ok(())
}


fn get_sine_wave(ampl: f32, phase_acc: f32, phase_acc_next: f32, mtu_size: usize) -> Vec<Complex<f32>> {
    let mut wave = Vec::new();
    let n_iter = mtu_size as i32;
    let mtu_size: f32 = FromPrimitive::from_usize(mtu_size).unwrap();
    let step_size: f32 = (phase_acc_next - phase_acc) / mtu_size;
    let mut phase: f32 = phase_acc;
    for _ in 0..n_iter{
        let sample: Complex<f32> = ampl * Complex::new(0., phase).exp();
        wave.push(sample);
        phase += step_size;
    }
    return wave
}


fn main() {

    let mut devices = soapysdr::enumerate("").unwrap();
    if devices.len() == 0 {panic!("No devices found, exiting...")}

    for dev in &devices {
        println!("Available devices:");
        println!("{}", dev);
    }

    let dev_args = devices.remove(0);
    println!("Using device: {}", dev_args);

    // TODO move params to argparse when done
    let samp_rate: f32 = 1e6;
    let bandwidth: f64 = 5e6;
    let freq: f64 = 1e9;

    let tx_channel: usize = 0;
    let tx_antenna: &str =  "BAND1";
    let tx_gain: f64 = 30.0;
    let tx_args: Args = "".into();
    // let tx_args: Args = "LO_frequency=434e6".into();
    // tx_args.set("set_LO_frequency", "434e6");
    let rx_channel: usize = 0;
    let rx_antenna: &str =  "LNAW";
    let rx_gain: f64 = 10.0;
    let rx_args: Args = "".into();


    // TODO factorise param setting
    let sdr = soapysdr::Device::new(dev_args).unwrap();

    sdr.set_sample_rate(Direction::Tx, tx_channel, samp_rate.into()).expect("Cannot set sample rate");
    println!("Actual Tx rate (Msps): {}", sdr.sample_rate(Direction::Tx, tx_channel).unwrap() / 1e6);
    sdr.set_bandwidth(Direction::Tx, tx_channel, bandwidth).expect("Cannot set bandwidth");
    println!("Actual Tx bandwidth (MHz): {}", sdr.bandwidth(Direction::Tx, tx_channel).unwrap() / 1e6);
    sdr.set_antenna(Direction::Tx, tx_channel, tx_antenna).expect("Cannot set antenna");
    println!("Actual Tx antenna: {}", sdr.antenna(Direction::Tx, tx_channel).unwrap());
    sdr.set_gain(Direction::Tx, tx_channel, tx_gain).expect("Cannot set gain");
    println!("Actual Tx gain (dB): {}", sdr.gain(Direction::Tx, tx_channel).unwrap());
    sdr.set_frequency(Direction::Tx, tx_channel, freq, tx_args).expect("Cannot set frequency");
    println!("Actual Tx passband freq (MHz): {}", sdr.frequency(Direction::Tx, tx_channel).unwrap() / 1e6);

    let mut tx_stream: TxStream<Complex<f32>> = sdr.tx_stream(&[tx_channel]).expect("Fail to initialize Tx");
    let tx_mtu: usize = tx_stream.mtu().expect("Fail to read Tx MTU");
    let tx_mtu_float: f32 = FromPrimitive::from_usize(tx_mtu).unwrap();
    println!("Tx MTU: {} elements", tx_mtu);

    sdr.set_sample_rate(Direction::Rx, rx_channel, samp_rate.into()).expect("Cannot set sample rate");
    println!("Actual Rx rate (Msps): {}", sdr.sample_rate(Direction::Rx, rx_channel).unwrap() / 1e6);
    sdr.set_bandwidth(Direction::Rx, rx_channel, bandwidth).expect("Cannot set bandwidth");
    println!("Actual Rx bandwidth (MHz): {}", sdr.bandwidth(Direction::Rx, rx_channel).unwrap() / 1e6);
    sdr.set_antenna(Direction::Rx, rx_channel, rx_antenna).expect("Cannot set antenna");
    println!("Actual Rx antenna: {}", sdr.antenna(Direction::Rx, rx_channel).unwrap());
    sdr.set_gain(Direction::Rx, rx_channel, rx_gain).expect("Cannot set gain");
    println!("Actual Rx gain (dB): {}", sdr.gain(Direction::Rx, rx_channel).unwrap());
    sdr.set_frequency(Direction::Rx, rx_channel, freq, rx_args).expect("Cannot set frequency");
    println!("Actual Rx passband freq (MHz): {}", sdr.frequency(Direction::Rx, rx_channel).unwrap() / 1e6);

    let mut rx_stream: RxStream<Complex<f32>> = sdr.rx_stream(&[rx_channel]).expect("Fail to initialize Tx");
    let rx_mtu: usize = rx_stream.mtu().expect("Fail to read Rx MTU");
    let rx_mtu_float: f32 = FromPrimitive::from_usize(rx_mtu).unwrap();
    println!("Rx MTU: {} elements", rx_mtu);

    println!("Letting things settle");
    thread::sleep(time::Duration::from_secs(1));

    tx_stream.activate(None).expect("Fail to activate Tx stream");

    let pi: f32 = FloatConst::PI();
    let mut phase_acc: f32 = 0.;
    let signal_ampl: f32 = 0.7;
    let wave_freq = samp_rate / 20.0;
    let phase_inc: f32 = 2. * pi * wave_freq / samp_rate;  // how much the phase increases per sample
    let mut phase_acc_next = phase_acc + tx_mtu_float * phase_inc;

    let now = Instant::now();
    let sdr_time: i64 = sdr.get_hardware_time(None).expect("Unable to get hardware time");
    let mut tx_start_time: Option<i64> = Some(sdr_time + (1e8 as i64));  // 100ms in nanosec notation
    let rx_start_time: Option<i64> = Some(sdr_time + (1e8 as i64) - ((rx_mtu_float / samp_rate) as i64) * (5e8 as i64));
    rx_stream.activate(rx_start_time).expect("Fail to activate Rx stream");

    let mut counter: u64 = 0;
    // let mut received_signal = vec![Complex::new(0., 0.); rx_mtu];
    while now.elapsed().as_secs() < 1 {
        if counter > 0 {
            tx_start_time = None;
        }
        let wave = get_sine_wave(signal_ampl, phase_acc, phase_acc_next, tx_mtu);
        let samples_written = tx_stream.write(&[&wave], tx_start_time, false, 100000).expect("Failed to write samples");
        assert_eq!(samples_written, tx_mtu);

        let mut rx_buffer = vec![Complex::new(0., 0.); rx_mtu];
        let read_len = rx_stream.read(&[&mut rx_buffer], 100000).expect("Rx read failed");
        // assert_eq!(read_len, rx_mtu);
        let file_name= format!("rx_samples_{counter}.cfile");
        let mut outfile = BufWriter::new(File::create(file_name).expect("error opening output file"));
        write_cfile(&rx_buffer[..read_len], &mut outfile).unwrap();

        phase_acc = phase_acc_next;
        phase_acc_next = phase_acc + tx_mtu_float * phase_inc;
        counter += 1;

    }

    tx_stream.deactivate(None).expect("Fail to deactivate Tx stream");
    rx_stream.deactivate(None).expect("Fail to deactivate Rx stream");

    // let dev_args = sdr.frequency_args_info(Direction::Tx, tx_channel).unwrap();
    // for d_args in &dev_args {
    //     println!("{}:{}", d_args.key, d_args.value);
    // }
    // println!("{:?}", sdr.frequency_range(Direction::Tx, 0).unwrap());
    // println!("driver key: {}", sdr.driver_key().unwrap());
    // println!("n channels: {}", sdr.num_channels(Direction::Tx).unwrap());
    // println!("channel info: {}", sdr.channel_info(Direction::Tx, 0).unwrap());
    // println!("hardware_info: {}", sdr.hardware_info().unwrap());
    // println!("full duplex: {}", sdr.full_duplex(Direction::Tx, 1).unwrap());
    // println!("antennas: {:?}", sdr.antennas(Direction::Tx, 1).unwrap());
}
