plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.qualcomm.qidk.inpaint"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.qualcomm.qidk.inpaint.v5"
        minSdk = 29
        targetSdk = 34
        versionCode = 6
        versionName = "5.0-MagicTouch"
        ndk {
            abiFilters.addAll(setOf("arm64-v8a"))
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    kotlinOptions {
        jvmTarget = "21"
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("com.github.jenly1314.WeChatQRCode:opencv:2.2.0")
    implementation("com.github.jenly1314.WeChatQRCode:opencv-armv64:2.2.0")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}
