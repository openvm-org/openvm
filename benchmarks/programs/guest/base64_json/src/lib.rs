use core::fmt::Debug;

use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct UserProfile<'a> {
    user_id: u64,
    username: &'a str,
    email: &'a str,
    full_name: &'a str,
    account_status: &'a str,
    address: Address<'a>,
    settings: Settings<'a>,
    preferences: Preferences,
    work_info: WorkInfo<'a>,
    social_links: SocialLinks<'a>,
    devices: [Device<'a>; 3],
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct Address<'a> {
    street: &'a str,
    city: &'a str,
    state: &'a str,
    zip: &'a str,
    country: &'a str,
    latitude: f64,
    longitude: f64,
    timezone: &'a str,
    region: &'a str,
    locality: &'a str,
    building: &'a str,
    apartment: &'a str,
    landmark: &'a str,
    floor_number: u8,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct Settings<'a> {
    theme: &'a str,
    font_size: u8,
    dark_mode: bool,
    email_notifications: bool,
    sms_notifications: bool,
    app_notifications: bool,
    password_strength: &'a str,
    login_alerts: bool,
    ad_preferences: AdPreferences,
    security: SecuritySettings<'a>,
    language: &'a str,
    privacy_mode: bool,
    location_sharing: bool,
    auto_backup: bool,
    timezone_auto_detect: bool,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct SecuritySettings<'a> {
    two_factor_auth: bool,
    recovery_email: &'a str,
    login_alerts: bool,
    trusted_devices: u32,
    last_password_change: &'a str,
    account_lock_enabled: bool,
    security_questions_enabled: bool,
    last_security_update: &'a str,
    last_failed_login: &'a str,
    password_expiration_days: u16,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct AdPreferences {
    personalized_ads: bool,
    third_party_sharing: bool,
    location_based_ads: bool,
    email_promotions: bool,
    sms_promotions: bool,
    web_tracking: bool,
    ad_frequency: u8,
    ad_relevance: f32,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct Preferences {
    color_scheme: u8,
    layout: u8,
    font_style: u8,
    text_size: u8,
    widget_visibility: bool,
    show_tutorials: bool,
    quick_access_enabled: bool,
    high_contrast_mode: bool,
    notification_sound: u8,
    vibration_feedback: bool,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct WorkInfo<'a> {
    position: &'a str,
    department: &'a str,
    company: &'a str,
    years_of_experience: u16,
    office_location: &'a str,
    employee_id: &'a str,
    manager_name: &'a str,
    team_name: &'a str,
    office_floor: u8,
    desk_number: u16,
    remote_work_enabled: bool,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct SocialLinks<'a> {
    twitter: &'a str,
    linkedin: &'a str,
    github: &'a str,
    website: &'a str,
    facebook: &'a str,
    instagram: &'a str,
    medium: &'a str,
    youtube: &'a str,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct Device<'a> {
    device_name: &'a str,
    os: &'a str,
    os_version: &'a str,
    last_seen: &'a str,
    gps_enabled: bool,
    bluetooth_enabled: bool,
    camera_access: bool,
    microphone_access: bool,
    network_type: &'a str,
    battery_level: u8,
    screen_brightness: u8,
    storage_capacity: u32,
    storage_used: u32,
    location_services: bool,
}
