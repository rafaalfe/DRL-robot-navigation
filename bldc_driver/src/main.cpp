/*
 * =================================================================================
 * KODE UJI OPEN-LOOP: MEREKAM RESPON MOTOR ALAMI (DENGAN KONTROL TELEOP)
 * =================================================================================
 * Deskripsi:
 * Versi kode ini dimodifikasi untuk pengujian "open-loop" menggunakan teleop.
 * PID dinonaktifkan. Perintah dari /cmd_vel akan diinterpretasikan secara berbeda:
 * `linear.x` akan digunakan sebagai NILAI PWM LANGSUNG untuk motor.
 *
 * CARA MENGGUNAKAN:
 * 1. Atur OPEN_LOOP_TEST_MODE menjadi 'true'.
 * 2. Upload kode ini ke ESP32.
 * 3. Gunakan teleop (keyboard, joystick, atau rostopic pub) untuk mengirim pesan
 * ke topik /cmd_vel.
 * 4. Contoh: Kirim pesan dengan `linear.x: 150` untuk menjalankan motor dengan PWM 150.
 * 5. Kirim pesan dengan `linear.x: 0` untuk menghentikan motor.
 * 6. Rekam data dari topik `/motor_rpms_actual` untuk dianalisis.
 * =================================================================================
 */

// --- 1. Termasuk Library ---
#include <ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Vector3.h>
#include <PID_v1.h>

// --- MODIFIKASI OPEN-LOOP: Saklar untuk mengaktifkan mode uji ---
// true = Mode Uji Open-Loop (PID mati, PWM dikontrol via teleop)
// false = Mode Normal (Kontrol PID aktif)
const bool OPEN_LOOP_TEST_MODE = false;

// --- 2. Definisi Pinout ---
const int PWM_PIN_L = 25;
const int DIR_PIN_L = 33;
const int ENABLE_PIN_L = 32;
const int RPM_PIN_L = 19;
const int CURRENT_PIN_L = 34;

const int PWM_PIN_R = 26;
const int DIR_PIN_R = 27;
const int ENABLE_PIN_R = 14;
const int RPM_PIN_R = 18;
const int CURRENT_PIN_R = 35;

// --- 3. Konfigurasi & Parameter ---
const float WHEEL_DIAMETER = 0.165;
const float WHEEL_SEPARATION = 0.40;
const int PULSES_PER_REVOLUTION = 90;

const float MAX_CURRENT_LIMIT = 4.0;
const int PWM_MAX = 250;
const int PWM_FREQ = 2000;
const int PWM_CHANNEL_L = 0;
const int PWM_CHANNEL_R = 1;
const int PWM_RESOLUTION = 8;

const int KICKSTART_PWM = 40;
const int KICKSTART_DURATION = 15;

const float ACS712_SENSITIVITY = 0.066;
int adc_zero_point_L = 2048;
int adc_zero_point_R = 2048;

const float RPM_FILTER_ALPHA = 0.1; 

// --- 4. Inisialisasi Objek & Variabel Global ---
ros::NodeHandle nh;
unsigned long last_cmd_vel_time = 0;
void cmdVelCallback(const geometry_msgs::Twist& msg);
ros::Subscriber<geometry_msgs::Twist> sub("/cmd_vel", &cmdVelCallback);

geometry_msgs::Vector3 current_msg;
ros::Publisher current_pub("motor_currents", &current_msg);
unsigned long last_current_pub_time = 0;

geometry_msgs::Vector3 rpm_actual_msg;
ros::Publisher rpm_actual_pub("motor_rpms_actual", &rpm_actual_msg);
unsigned long last_rpm_pub_time = 0;

geometry_msgs::Vector3 rpm_target_msg;
ros::Publisher rpm_target_pub("debug/motor_rpms_target", &rpm_target_msg);
geometry_msgs::Vector3 pid_output_msg;
ros::Publisher pid_output_pub("debug/pid_outputs", &pid_output_msg);
geometry_msgs::Vector3 final_pwm_msg;
ros::Publisher final_pwm_pub("debug/final_pwms", &final_pwm_msg);
unsigned long last_debug_pub_time = 0;

bool ros_command_received = false;

double target_rpm_L = 0.0;
double target_rpm_R = 0.0;
double actual_rpm_L = 0.0;
double actual_rpm_R = 0.0;
double filtered_rpm_L = 0.0;
double filtered_rpm_R = 0.0;
double prev_target_rpm_L = 0.0;
double prev_target_rpm_R = 0.0;

// --- MODIFIKASI OPEN-LOOP: Variabel untuk menyimpan PWM dari teleop ---
double open_loop_pwm = 0.0;

volatile long pulse_count_L = 0;
volatile long pulse_count_R = 0;
unsigned long last_rpm_calc_time = 0;

double Kp = 0.56323, Ki = 1.9673, Kd = 0.0054356;
double pid_output_L, pid_output_R;

PID pid_L(&filtered_rpm_L, &pid_output_L, &target_rpm_L, Kp, Ki, Kd, DIRECT);
PID pid_R(&filtered_rpm_R, &pid_output_R, &target_rpm_R, Kp, Ki, Kd, DIRECT);

// --- 5. Fungsi-fungsi ---
void IRAM_ATTR pulseCounterL() { pulse_count_L++; }
void IRAM_ATTR pulseCounterR() { pulse_count_R++; }

void cmdVelCallback(const geometry_msgs::Twist& msg) {
    if (!ros_command_received) {
        ros_command_received = true;
        nh.loginfo("Perintah /cmd_vel pertama diterima.");
    }

    if (OPEN_LOOP_TEST_MODE) {
        // --- MODE UJI OPEN-LOOP ---
        // Gunakan linear.x dari teleop sebagai nilai PWM langsung
        open_loop_pwm = msg.linear.x;

        // Batasi nilai PWM untuk keamanan
        if (open_loop_pwm < 0) open_loop_pwm = 0;
        if (open_loop_pwm > PWM_MAX) open_loop_pwm = PWM_MAX;
        
    } else {
        // --- MODE NORMAL (PID) ---
        float linear_vel = msg.linear.x;
        float angular_vel = -msg.angular.z;
        float vel_R = (2.0 * linear_vel + angular_vel * WHEEL_SEPARATION) / 2.0;
        float vel_L = (2.0 * linear_vel - angular_vel * WHEEL_SEPARATION) / 2.0;
        float wheel_circumference = PI * WHEEL_DIAMETER;
        target_rpm_R = (vel_R / wheel_circumference) * 60.0;
        target_rpm_L = (vel_L / wheel_circumference) * 60.0;
    }
    last_cmd_vel_time = millis();
}

float readCurrent(int pin, int zero_point) {
    long total_adc_val = 0;
    for (int i = 0; i < 10; i++) { total_adc_val += analogRead(pin); }
    float avg_adc_val = total_adc_val / 10.0;
    float adc_voltage = avg_adc_val * (3.3 / 4095.0);
    const float VOLTAGE_DIVIDER_FACTOR = 1.6091;
    float sensor_voltage = adc_voltage * VOLTAGE_DIVIDER_FACTOR;
    float zero_point_voltage = (float)zero_point * (3.3 / 4095.0) * VOLTAGE_DIVIDER_FACTOR;
    return abs((sensor_voltage - zero_point_voltage) / ACS712_SENSITIVITY);
}

void moveMotor(int pwm, int pwm_channel, int dir_pin, double prev_target_rpm, double current_target_rpm) {
    if (pwm >= 0) {
        if (dir_pin == DIR_PIN_L) { digitalWrite(dir_pin, LOW); } 
        else { digitalWrite(dir_pin, HIGH); }
    } else {
        if (dir_pin == DIR_PIN_L) { digitalWrite(dir_pin, HIGH); } 
        else { digitalWrite(dir_pin, LOW); }
    }
    if (prev_target_rpm == 0 && current_target_rpm != 0) {
        ledcWrite(pwm_channel, KICKSTART_PWM);
        delay(KICKSTART_DURATION);
    }
    ledcWrite(pwm_channel, abs(pwm));
}

// --- 6. Fungsi Setup Utama ---
void setup() {
    Serial.begin(57600);
    pinMode(DIR_PIN_L, OUTPUT); pinMode(ENABLE_PIN_L, OUTPUT);
    pinMode(DIR_PIN_R, OUTPUT); pinMode(ENABLE_PIN_R, OUTPUT);
    pinMode(RPM_PIN_L, INPUT_PULLUP); pinMode(RPM_PIN_R, INPUT_PULLUP);
    digitalWrite(ENABLE_PIN_L, HIGH); digitalWrite(ENABLE_PIN_R, HIGH);
    
    ledcSetup(PWM_CHANNEL_L, PWM_FREQ, PWM_RESOLUTION);
    ledcAttachPin(PWM_PIN_L, PWM_CHANNEL_L);
    ledcSetup(PWM_CHANNEL_R, PWM_FREQ, PWM_RESOLUTION);
    ledcAttachPin(PWM_PIN_R, PWM_CHANNEL_R);
    ledcWrite(PWM_CHANNEL_L, 0); ledcWrite(PWM_CHANNEL_R, 0);

    nh.initNode();
    nh.logwarn("Kalibrasi sensor arus...");
    delay(2000);
    long total_adc_L = 0, total_adc_R = 0;
    for (int i = 0; i < 500; i++) {
        total_adc_L += analogRead(CURRENT_PIN_L);
        total_adc_R += analogRead(CURRENT_PIN_R);
        delay(2);
    }
    adc_zero_point_L = total_adc_L / 500; adc_zero_point_R = total_adc_R / 500;
    nh.logwarn("Kalibrasi selesai.");

    // Berlangganan ke /cmd_vel di kedua mode
    nh.subscribe(sub);
    
    if (OPEN_LOOP_TEST_MODE) {
        nh.logwarn("!!! MODE UJI OPEN-LOOP AKTIF !!!");
        nh.logwarn("PID dinonaktifkan. Kontrol PWM via /cmd_vel (linear.x).");
    } else {
        pid_L.SetMode(AUTOMATIC);
        pid_R.SetMode(AUTOMATIC);
        pid_L.SetSampleTime(20);
        pid_R.SetSampleTime(20);
        pid_L.SetOutputLimits(-PWM_MAX, PWM_MAX);
        pid_R.SetOutputLimits(-PWM_MAX, PWM_MAX);
        nh.logwarn("Mode Normal (PID) aktif. Menunggu perintah /cmd_vel...");
    }
    
    // Publisher tetap diaktifkan untuk merekam data
    nh.advertise(current_pub);
    nh.advertise(rpm_actual_pub);
    nh.advertise(rpm_target_pub);
    nh.advertise(pid_output_pub);
    nh.advertise(final_pwm_pub);
    
    attachInterrupt(digitalPinToInterrupt(RPM_PIN_L), pulseCounterL, RISING);
    attachInterrupt(digitalPinToInterrupt(RPM_PIN_R), pulseCounterR, RISING);
}

void loop() {
    nh.spinOnce();
    unsigned long now = millis();

    // =================================================================
    // LANGKAH 1: SELALU BACA SENSOR DAN UPDATE STATE AKTUAL
    // =================================================================
    if (now - last_rpm_calc_time >= 100) {
        noInterrupts();
        long pulses_L = pulse_count_L; long pulses_R = pulse_count_R;
        pulse_count_L = 0; pulse_count_R = 0;
        interrupts();
        double delta_time_sec = (now - last_rpm_calc_time) / 1000.0;
        if (delta_time_sec > 0) {
            actual_rpm_L = (pulses_L / (double)PULSES_PER_REVOLUTION) / delta_time_sec * 60.0;
            actual_rpm_R = (pulses_R / (double)PULSES_PER_REVOLUTION) / delta_time_sec * 60.0;
        } else {
            actual_rpm_L = 0;
            actual_rpm_R = 0;
        }

        if (!OPEN_LOOP_TEST_MODE) {
            if (target_rpm_L < 0) { actual_rpm_L *= -1.0; }
            if (target_rpm_R < 0) { actual_rpm_R *= -1.0; }
        }
        
        last_rpm_calc_time = now;
        filtered_rpm_L = (RPM_FILTER_ALPHA * actual_rpm_L) + (1.0 - RPM_FILTER_ALPHA) * filtered_rpm_L;
        filtered_rpm_R = (RPM_FILTER_ALPHA * actual_rpm_R) + (1.0 - RPM_FILTER_ALPHA) * filtered_rpm_R;
    }

    // =================================================================
    // LANGKAH 2 & 3: EKSEKUSI KONTROL
    // =================================================================
    if (OPEN_LOOP_TEST_MODE) {
        // === MODE UJI OPEN-LOOP (DENGAN TELEOP) ===
        if (open_loop_pwm > 0) {
            digitalWrite(DIR_PIN_L, LOW);
            digitalWrite(DIR_PIN_R, HIGH);
            ledcWrite(PWM_CHANNEL_L, open_loop_pwm);
            ledcWrite(PWM_CHANNEL_R, open_loop_pwm);
        } else {
            ledcWrite(PWM_CHANNEL_L, 0);
            ledcWrite(PWM_CHANNEL_R, 0);
        }

    } else {
        // === MODE NORMAL (KONTROL PID) ===
        const double RPM_DEADBAND = 0.1; 
        if (!ros_command_received || (abs(target_rpm_L) < RPM_DEADBAND && abs(target_rpm_R) < RPM_DEADBAND)) {
            ledcWrite(PWM_CHANNEL_L, 0);
            ledcWrite(PWM_CHANNEL_R, 0);
            pid_output_L = 0;
            pid_output_R = 0;
            pid_L.SetMode(MANUAL);
            pid_R.SetMode(MANUAL);
            prev_target_rpm_L = 0;
            prev_target_rpm_R = 0;
        } 
        else {
            pid_L.SetMode(AUTOMATIC);
            pid_R.SetMode(AUTOMATIC);
            pid_L.Compute();
            pid_R.Compute();

            long final_pwm_L = pid_output_L;
            long final_pwm_R = pid_output_R;
            
            float current_L = readCurrent(CURRENT_PIN_L, adc_zero_point_L);
            float current_R = readCurrent(CURRENT_PIN_R, adc_zero_point_R);
            if (current_L > MAX_CURRENT_LIMIT) { final_pwm_L = 0; nh.logerror("Arus motor KIRI berlebih!"); }
            if (current_R > MAX_CURRENT_LIMIT) { final_pwm_R = 0; nh.logerror("Arus motor KANAN berlebih!"); }

            moveMotor(final_pwm_L, PWM_CHANNEL_L, DIR_PIN_L, prev_target_rpm_L, target_rpm_L);
            moveMotor(final_pwm_R, PWM_CHANNEL_R, DIR_PIN_R, prev_target_rpm_R, target_rpm_R);

            prev_target_rpm_L = target_rpm_L;
            prev_target_rpm_R = target_rpm_R;
        }
    }

    // =================================================================
    // LANGKAH 4: SELALU PUBLIKASIKAN DATA TERBARU
    // =================================================================
    if (now - last_current_pub_time >= 100) {
        current_msg.x = readCurrent(CURRENT_PIN_L, adc_zero_point_L);
        current_msg.y = readCurrent(CURRENT_PIN_R, adc_zero_point_R);
        current_pub.publish(&current_msg);
        last_current_pub_time = now;
    }

    if (now - last_rpm_pub_time >= 100) {
        rpm_actual_msg.x = actual_rpm_L; 
        rpm_actual_msg.y = actual_rpm_R;
        rpm_actual_pub.publish(&rpm_actual_msg);
        last_rpm_pub_time = now;
    }
    
    if (now - last_debug_pub_time >= 100) {
        rpm_target_msg.x = OPEN_LOOP_TEST_MODE ? 0 : target_rpm_L;
        rpm_target_msg.y = OPEN_LOOP_TEST_MODE ? 0 : target_rpm_R;
        rpm_target_pub.publish(&rpm_target_msg);
        
        pid_output_msg.x = OPEN_LOOP_TEST_MODE ? 0 : pid_output_L;
        pid_output_msg.y = OPEN_LOOP_TEST_MODE ? 0 : pid_output_R;
        pid_output_pub.publish(&pid_output_msg);
        
        final_pwm_msg.x = OPEN_LOOP_TEST_MODE ? open_loop_pwm : pid_output_L;
        final_pwm_msg.y = OPEN_LOOP_TEST_MODE ? open_loop_pwm : pid_output_R;
        final_pwm_pub.publish(&final_pwm_msg);
        
        last_debug_pub_time = now;
    }

    delay(10);
}
