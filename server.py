from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, UploadFile, File, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import bcrypt
import re
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Settings
SECRET_KEY = os.environ.get('JWT_SECRET', 'knowledgedottcom-secret-key-2024')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# Create the main app
app = FastAPI(
    title="KnowledgeDottCom API",
    description="Backend API for KnowledgeDottCom Educational Platform",
    version="1.0.0"
)

# Create router with /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer(auto_error=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

# Base Models
class BaseDBModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# User Models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    phone: Optional[str] = None
    user_type: str  # student, teacher, organization

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseDBModel, UserBase):
    is_active: bool = True
    is_verified: bool = False
    profile_image: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    phone: Optional[str] = None
    user_type: str
    is_active: bool
    is_verified: bool
    profile_image: Optional[str] = None
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Teacher Profile Models
class TeacherProfileCreate(BaseModel):
    bio: Optional[str] = None
    subjects: List[str] = []
    boards: List[str] = []  # AQA, Edexcel, OCR, CIE, IB, etc.
    qualifications: List[str] = []
    experience_years: int = 0
    hourly_rate: float = 0
    currency: str = "GBP"
    teaching_modes: List[str] = []  # online, home, both
    location: Optional[str] = None
    availability: Optional[dict] = None

class TeacherProfile(BaseDBModel):
    user_id: str
    bio: Optional[str] = None
    subjects: List[str] = []
    boards: List[str] = []
    qualifications: List[str] = []
    experience_years: int = 0
    hourly_rate: float = 0
    currency: str = "GBP"
    teaching_modes: List[str] = []
    location: Optional[str] = None
    availability: Optional[dict] = None
    rating: float = 0
    total_reviews: int = 0
    total_students: int = 0
    is_approved: bool = False

class TeacherProfileResponse(BaseModel):
    id: str
    user_id: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    profile_image: Optional[str] = None
    bio: Optional[str] = None
    subjects: List[str] = []
    boards: List[str] = []
    qualifications: List[str] = []
    experience_years: int = 0
    hourly_rate: float = 0
    currency: str = "GBP"
    teaching_modes: List[str] = []
    location: Optional[str] = None
    rating: float = 0
    total_reviews: int = 0
    is_approved: bool = False

# Student Profile Models
class StudentProfileCreate(BaseModel):
    grade_level: Optional[str] = None
    boards: List[str] = []
    subjects_interested: List[str] = []
    learning_goals: Optional[str] = None
    parent_name: Optional[str] = None
    parent_phone: Optional[str] = None

class StudentProfile(BaseDBModel):
    user_id: str
    grade_level: Optional[str] = None
    boards: List[str] = []
    subjects_interested: List[str] = []
    learning_goals: Optional[str] = None
    parent_name: Optional[str] = None
    parent_phone: Optional[str] = None

# Organization Profile Models
class OrganizationProfileCreate(BaseModel):
    organization_name: str
    organization_type: str  # school, college, training_center, corporate
    address: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    requirements: List[str] = []

class OrganizationProfile(BaseDBModel):
    user_id: str
    organization_name: str
    organization_type: str
    address: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    requirements: List[str] = []
    is_verified: bool = False

# Course/Resource Models
class BoardCategory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # AQA, Edexcel, OCR A, etc.
    category: str  # uk, international
    subjects: List[str] = []
    description: Optional[str] = None

class CourseCreate(BaseModel):
    title: str
    description: str
    board: str  # AQA, Edexcel, CIE, etc.
    board_category: str  # uk, international
    subject: str
    level: str  # GCSE, A-Level, etc.
    topics: List[str] = []
    is_free: bool = True
    price: float = 0
    currency: str = "GBP"

class Course(BaseDBModel):
    teacher_id: str
    title: str
    description: str
    board: str
    board_category: str
    subject: str
    level: str
    topics: List[str] = []
    is_free: bool = True
    price: float = 0
    currency: str = "GBP"
    thumbnail: Optional[str] = None
    total_lessons: int = 0
    total_enrolled: int = 0
    rating: float = 0
    is_published: bool = False

class CourseResponse(BaseModel):
    id: str
    teacher_id: str
    teacher_name: Optional[str] = None
    title: str
    description: str
    board: str
    board_category: str
    subject: str
    level: str
    topics: List[str] = []
    is_free: bool
    price: float
    currency: str
    thumbnail: Optional[str] = None
    total_lessons: int
    total_enrolled: int
    rating: float
    is_published: bool
    created_at: datetime

# Lesson Models
class LessonCreate(BaseModel):
    title: str
    description: Optional[str] = None
    content_type: str  # video, document, quiz, assignment
    content_url: Optional[str] = None
    content_text: Optional[str] = None
    duration_minutes: int = 0
    order: int = 0

class Lesson(BaseDBModel):
    course_id: str
    title: str
    description: Optional[str] = None
    content_type: str
    content_url: Optional[str] = None
    content_text: Optional[str] = None
    duration_minutes: int = 0
    order: int = 0
    is_free_preview: bool = False

# Enrollment Models
class EnrollmentCreate(BaseModel):
    course_id: str

class Enrollment(BaseDBModel):
    user_id: str
    course_id: str
    progress: float = 0
    completed_lessons: List[str] = []
    is_completed: bool = False

# Booking Models
class BookingCreate(BaseModel):
    teacher_id: str
    subject: str
    booking_type: str  # lesson, assignment, project, quiz_help
    scheduled_date: datetime
    duration_minutes: int = 60
    mode: str  # online, home
    notes: Optional[str] = None

class Booking(BaseDBModel):
    student_id: str
    teacher_id: str
    subject: str
    booking_type: str
    scheduled_date: datetime
    duration_minutes: int = 60
    mode: str
    notes: Optional[str] = None
    status: str = "pending"  # pending, confirmed, completed, cancelled
    price: float = 0
    currency: str = "GBP"

class BookingResponse(BaseModel):
    id: str
    student_id: str
    student_name: Optional[str] = None
    teacher_id: str
    teacher_name: Optional[str] = None
    subject: str
    booking_type: str
    scheduled_date: datetime
    duration_minutes: int
    mode: str
    notes: Optional[str] = None
    status: str
    price: float
    currency: str
    created_at: datetime

# Job Models
class JobCreate(BaseModel):
    title: str
    description: str
    job_type: str  # online_tutoring, home_tutoring, school_teaching, assignment
    subjects: List[str] = []
    boards: List[str] = []
    location: Optional[str] = None
    salary_range: Optional[str] = None
    requirements: List[str] = []

class Job(BaseDBModel):
    posted_by: str  # organization or admin
    title: str
    description: str
    job_type: str
    subjects: List[str] = []
    boards: List[str] = []
    location: Optional[str] = None
    salary_range: Optional[str] = None
    requirements: List[str] = []
    is_active: bool = True
    applications_count: int = 0

# Review Models
class ReviewCreate(BaseModel):
    teacher_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None

class Review(BaseDBModel):
    student_id: str
    teacher_id: str
    rating: int
    comment: Optional[str] = None

# Contact/Inquiry Models
class InquiryCreate(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    subject: str
    message: str
    inquiry_type: str  # student, teacher, organization, general

class Inquiry(BaseDBModel):
    name: str
    email: str
    phone: Optional[str] = None
    subject: str
    message: str
    inquiry_type: str
    status: str = "new"  # new, in_progress, resolved

# Search Models
class SearchQuery(BaseModel):
    subject: Optional[str] = None
    location: Optional[str] = None
    board: Optional[str] = None
    teaching_mode: Optional[str] = None
    min_rate: Optional[float] = None
    max_rate: Optional[float] = None

# ==================== HELPER FUNCTIONS ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(user_id: str, user_type: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "user_type": user_type,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    payload = decode_token(credentials.credentials)
    user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

async def get_optional_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    if not credentials:
        return None
    try:
        payload = decode_token(credentials.credentials)
        user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0})
        return user
    except:
        return None

def serialize_datetime(obj):
    """Convert datetime to ISO string for MongoDB storage"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def serialize_doc(doc: dict) -> dict:
    """Serialize document for MongoDB storage"""
    serialized = {}
    for key, value in doc.items():
        serialized[key] = serialize_datetime(value)
    return serialized

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    # Check if email exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate user type
    if user_data.user_type not in ["student", "teacher", "organization"]:
        raise HTTPException(status_code=400, detail="Invalid user type")
    
    # Create user
    user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        phone=user_data.phone,
        user_type=user_data.user_type
    )
    
    user_dict = serialize_doc(user.model_dump())
    user_dict["password_hash"] = hash_password(user_data.password)
    
    await db.users.insert_one(user_dict)
    
    # Create empty profile based on user type
    if user_data.user_type == "teacher":
        profile = TeacherProfile(user_id=user.id)
        await db.teacher_profiles.insert_one(serialize_doc(profile.model_dump()))
    elif user_data.user_type == "student":
        profile = StudentProfile(user_id=user.id)
        await db.student_profiles.insert_one(serialize_doc(profile.model_dump()))
    elif user_data.user_type == "organization":
        profile = OrganizationProfile(
            user_id=user.id,
            organization_name=user_data.full_name,
            organization_type="other"
        )
        await db.organization_profiles.insert_one(serialize_doc(profile.model_dump()))
    
    token = create_access_token(user.id, user.user_type)
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            phone=user.phone,
            user_type=user.user_type,
            is_active=user.is_active,
            is_verified=user.is_verified,
            profile_image=user.profile_image,
            created_at=user.created_at
        )
    )

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not verify_password(credentials.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated")
    
    token = create_access_token(user["id"], user["user_type"])
    
    # Parse datetime if string
    created_at = user.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            phone=user.get("phone"),
            user_type=user["user_type"],
            is_active=user.get("is_active", True),
            is_verified=user.get("is_verified", False),
            profile_image=user.get("profile_image"),
            created_at=created_at
        )
    )

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    created_at = current_user.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        phone=current_user.get("phone"),
        user_type=current_user["user_type"],
        is_active=current_user.get("is_active", True),
        is_verified=current_user.get("is_verified", False),
        profile_image=current_user.get("profile_image"),
        created_at=created_at
    )

@api_router.put("/auth/update-profile")
async def update_user_profile(
    full_name: Optional[str] = None,
    phone: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if full_name:
        update_data["full_name"] = full_name
    if phone:
        update_data["phone"] = phone
    
    await db.users.update_one(
        {"id": current_user["id"]},
        {"$set": update_data}
    )
    return {"message": "Profile updated successfully"}

@api_router.put("/auth/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: dict = Depends(get_current_user)
):
    user = await db.users.find_one({"id": current_user["id"]}, {"_id": 0})
    if not verify_password(current_password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    await db.users.update_one(
        {"id": current_user["id"]},
        {"$set": {
            "password_hash": hash_password(new_password),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    return {"message": "Password changed successfully"}

# ==================== TEACHER ROUTES ====================

@api_router.get("/teachers", response_model=List[TeacherProfileResponse])
async def get_teachers(
    subject: Optional[str] = None,
    board: Optional[str] = None,
    location: Optional[str] = None,
    teaching_mode: Optional[str] = None,
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None,
    skip: int = 0,
    limit: int = 20
):
    """Get list of approved teachers with optional filters"""
    query = {"is_approved": True}
    
    if subject:
        query["subjects"] = {"$regex": subject, "$options": "i"}
    if board:
        query["boards"] = {"$regex": board, "$options": "i"}
    if location:
        query["location"] = {"$regex": location, "$options": "i"}
    if teaching_mode:
        query["teaching_modes"] = teaching_mode
    if min_rate is not None:
        query["hourly_rate"] = {"$gte": min_rate}
    if max_rate is not None:
        query.setdefault("hourly_rate", {})["$lte"] = max_rate
    
    profiles = await db.teacher_profiles.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
    
    # Enrich with user data
    result = []
    for profile in profiles:
        user = await db.users.find_one({"id": profile["user_id"]}, {"_id": 0})
        result.append(TeacherProfileResponse(
            id=profile["id"],
            user_id=profile["user_id"],
            full_name=user.get("full_name") if user else None,
            email=user.get("email") if user else None,
            profile_image=user.get("profile_image") if user else None,
            bio=profile.get("bio"),
            subjects=profile.get("subjects", []),
            boards=profile.get("boards", []),
            qualifications=profile.get("qualifications", []),
            experience_years=profile.get("experience_years", 0),
            hourly_rate=profile.get("hourly_rate", 0),
            currency=profile.get("currency", "GBP"),
            teaching_modes=profile.get("teaching_modes", []),
            location=profile.get("location"),
            rating=profile.get("rating", 0),
            total_reviews=profile.get("total_reviews", 0),
            is_approved=profile.get("is_approved", False)
        ))
    
    return result

@api_router.get("/teachers/{teacher_id}", response_model=TeacherProfileResponse)
async def get_teacher(teacher_id: str):
    """Get single teacher profile"""
    profile = await db.teacher_profiles.find_one({"id": teacher_id}, {"_id": 0})
    if not profile:
        # Try by user_id
        profile = await db.teacher_profiles.find_one({"user_id": teacher_id}, {"_id": 0})
    if not profile:
        raise HTTPException(status_code=404, detail="Teacher not found")
    
    user = await db.users.find_one({"id": profile["user_id"]}, {"_id": 0})
    
    return TeacherProfileResponse(
        id=profile["id"],
        user_id=profile["user_id"],
        full_name=user.get("full_name") if user else None,
        email=user.get("email") if user else None,
        profile_image=user.get("profile_image") if user else None,
        bio=profile.get("bio"),
        subjects=profile.get("subjects", []),
        boards=profile.get("boards", []),
        qualifications=profile.get("qualifications", []),
        experience_years=profile.get("experience_years", 0),
        hourly_rate=profile.get("hourly_rate", 0),
        currency=profile.get("currency", "GBP"),
        teaching_modes=profile.get("teaching_modes", []),
        location=profile.get("location"),
        rating=profile.get("rating", 0),
        total_reviews=profile.get("total_reviews", 0),
        is_approved=profile.get("is_approved", False)
    )

@api_router.put("/teachers/profile")
async def update_teacher_profile(
    profile_data: TeacherProfileCreate,
    current_user: dict = Depends(get_current_user)
):
    """Update teacher's own profile"""
    if current_user["user_type"] != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can update teacher profile")
    
    update_data = profile_data.model_dump()
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.teacher_profiles.update_one(
        {"user_id": current_user["id"]},
        {"$set": update_data}
    )
    return {"message": "Profile updated successfully"}

@api_router.get("/teachers/profile/me", response_model=TeacherProfileResponse)
async def get_my_teacher_profile(current_user: dict = Depends(get_current_user)):
    """Get current teacher's profile"""
    if current_user["user_type"] != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can access this")
    
    profile = await db.teacher_profiles.find_one({"user_id": current_user["id"]}, {"_id": 0})
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return TeacherProfileResponse(
        id=profile["id"],
        user_id=profile["user_id"],
        full_name=current_user.get("full_name"),
        email=current_user.get("email"),
        profile_image=current_user.get("profile_image"),
        bio=profile.get("bio"),
        subjects=profile.get("subjects", []),
        boards=profile.get("boards", []),
        qualifications=profile.get("qualifications", []),
        experience_years=profile.get("experience_years", 0),
        hourly_rate=profile.get("hourly_rate", 0),
        currency=profile.get("currency", "GBP"),
        teaching_modes=profile.get("teaching_modes", []),
        location=profile.get("location"),
        rating=profile.get("rating", 0),
        total_reviews=profile.get("total_reviews", 0),
        is_approved=profile.get("is_approved", False)
    )

# ==================== STUDENT ROUTES ====================

@api_router.put("/students/profile")
async def update_student_profile(
    profile_data: StudentProfileCreate,
    current_user: dict = Depends(get_current_user)
):
    """Update student's own profile"""
    if current_user["user_type"] != "student":
        raise HTTPException(status_code=403, detail="Only students can update student profile")
    
    update_data = profile_data.model_dump()
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.student_profiles.update_one(
        {"user_id": current_user["id"]},
        {"$set": update_data}
    )
    return {"message": "Profile updated successfully"}

@api_router.get("/students/profile/me")
async def get_my_student_profile(current_user: dict = Depends(get_current_user)):
    """Get current student's profile"""
    if current_user["user_type"] != "student":
        raise HTTPException(status_code=403, detail="Only students can access this")
    
    profile = await db.student_profiles.find_one({"user_id": current_user["id"]}, {"_id": 0})
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return profile

# ==================== ORGANIZATION ROUTES ====================

@api_router.put("/organizations/profile")
async def update_organization_profile(
    profile_data: OrganizationProfileCreate,
    current_user: dict = Depends(get_current_user)
):
    """Update organization's own profile"""
    if current_user["user_type"] != "organization":
        raise HTTPException(status_code=403, detail="Only organizations can update this profile")
    
    update_data = profile_data.model_dump()
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.organization_profiles.update_one(
        {"user_id": current_user["id"]},
        {"$set": update_data}
    )
    return {"message": "Profile updated successfully"}

@api_router.get("/organizations/profile/me")
async def get_my_organization_profile(current_user: dict = Depends(get_current_user)):
    """Get current organization's profile"""
    if current_user["user_type"] != "organization":
        raise HTTPException(status_code=403, detail="Only organizations can access this")
    
    profile = await db.organization_profiles.find_one({"user_id": current_user["id"]}, {"_id": 0})
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return profile

# ==================== COURSE ROUTES ====================

@api_router.post("/courses", response_model=CourseResponse)
async def create_course(
    course_data: CourseCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new course (teachers only)"""
    if current_user["user_type"] != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can create courses")
    
    course = Course(
        teacher_id=current_user["id"],
        **course_data.model_dump()
    )
    
    await db.courses.insert_one(serialize_doc(course.model_dump()))
    
    return CourseResponse(
        id=course.id,
        teacher_id=course.teacher_id,
        teacher_name=current_user.get("full_name"),
        title=course.title,
        description=course.description,
        board=course.board,
        board_category=course.board_category,
        subject=course.subject,
        level=course.level,
        topics=course.topics,
        is_free=course.is_free,
        price=course.price,
        currency=course.currency,
        thumbnail=course.thumbnail,
        total_lessons=course.total_lessons,
        total_enrolled=course.total_enrolled,
        rating=course.rating,
        is_published=course.is_published,
        created_at=course.created_at
    )

@api_router.get("/courses", response_model=List[CourseResponse])
async def get_courses(
    board: Optional[str] = None,
    board_category: Optional[str] = None,
    subject: Optional[str] = None,
    level: Optional[str] = None,
    is_free: Optional[bool] = None,
    teacher_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 20
):
    """Get list of published courses with filters"""
    query = {"is_published": True}
    
    if board:
        query["board"] = {"$regex": board, "$options": "i"}
    if board_category:
        query["board_category"] = board_category
    if subject:
        query["subject"] = {"$regex": subject, "$options": "i"}
    if level:
        query["level"] = {"$regex": level, "$options": "i"}
    if is_free is not None:
        query["is_free"] = is_free
    if teacher_id:
        query["teacher_id"] = teacher_id
    
    courses = await db.courses.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
    
    result = []
    for course in courses:
        teacher = await db.users.find_one({"id": course["teacher_id"]}, {"_id": 0})
        created_at = course.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        result.append(CourseResponse(
            id=course["id"],
            teacher_id=course["teacher_id"],
            teacher_name=teacher.get("full_name") if teacher else None,
            title=course["title"],
            description=course["description"],
            board=course["board"],
            board_category=course["board_category"],
            subject=course["subject"],
            level=course["level"],
            topics=course.get("topics", []),
            is_free=course.get("is_free", True),
            price=course.get("price", 0),
            currency=course.get("currency", "GBP"),
            thumbnail=course.get("thumbnail"),
            total_lessons=course.get("total_lessons", 0),
            total_enrolled=course.get("total_enrolled", 0),
            rating=course.get("rating", 0),
            is_published=course.get("is_published", False),
            created_at=created_at
        ))
    
    return result

@api_router.get("/courses/my", response_model=List[CourseResponse])
async def get_my_courses(current_user: dict = Depends(get_current_user)):
    """Get teacher's own courses"""
    if current_user["user_type"] != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can access this")
    
    courses = await db.courses.find({"teacher_id": current_user["id"]}, {"_id": 0}).to_list(100)
    
    result = []
    for course in courses:
        created_at = course.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        result.append(CourseResponse(
            id=course["id"],
            teacher_id=course["teacher_id"],
            teacher_name=current_user.get("full_name"),
            title=course["title"],
            description=course["description"],
            board=course["board"],
            board_category=course["board_category"],
            subject=course["subject"],
            level=course["level"],
            topics=course.get("topics", []),
            is_free=course.get("is_free", True),
            price=course.get("price", 0),
            currency=course.get("currency", "GBP"),
            thumbnail=course.get("thumbnail"),
            total_lessons=course.get("total_lessons", 0),
            total_enrolled=course.get("total_enrolled", 0),
            rating=course.get("rating", 0),
            is_published=course.get("is_published", False),
            created_at=created_at
        ))
    
    return result

@api_router.get("/courses/{course_id}", response_model=CourseResponse)
async def get_course(course_id: str):
    """Get single course details"""
    course = await db.courses.find_one({"id": course_id}, {"_id": 0})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    teacher = await db.users.find_one({"id": course["teacher_id"]}, {"_id": 0})
    created_at = course.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    return CourseResponse(
        id=course["id"],
        teacher_id=course["teacher_id"],
        teacher_name=teacher.get("full_name") if teacher else None,
        title=course["title"],
        description=course["description"],
        board=course["board"],
        board_category=course["board_category"],
        subject=course["subject"],
        level=course["level"],
        topics=course.get("topics", []),
        is_free=course.get("is_free", True),
        price=course.get("price", 0),
        currency=course.get("currency", "GBP"),
        thumbnail=course.get("thumbnail"),
        total_lessons=course.get("total_lessons", 0),
        total_enrolled=course.get("total_enrolled", 0),
        rating=course.get("rating", 0),
        is_published=course.get("is_published", False),
        created_at=created_at
    )

@api_router.put("/courses/{course_id}")
async def update_course(
    course_id: str,
    course_data: CourseCreate,
    current_user: dict = Depends(get_current_user)
):
    """Update a course"""
    course = await db.courses.find_one({"id": course_id}, {"_id": 0})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course["teacher_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to update this course")
    
    update_data = course_data.model_dump()
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.courses.update_one({"id": course_id}, {"$set": update_data})
    return {"message": "Course updated successfully"}

@api_router.put("/courses/{course_id}/publish")
async def publish_course(
    course_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Publish/unpublish a course"""
    course = await db.courses.find_one({"id": course_id}, {"_id": 0})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course["teacher_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    new_status = not course.get("is_published", False)
    await db.courses.update_one(
        {"id": course_id},
        {"$set": {"is_published": new_status, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": f"Course {'published' if new_status else 'unpublished'} successfully"}

@api_router.delete("/courses/{course_id}")
async def delete_course(
    course_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a course"""
    course = await db.courses.find_one({"id": course_id}, {"_id": 0})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course["teacher_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.courses.delete_one({"id": course_id})
    await db.lessons.delete_many({"course_id": course_id})
    return {"message": "Course deleted successfully"}

# ==================== LESSON ROUTES ====================

@api_router.post("/courses/{course_id}/lessons")
async def create_lesson(
    course_id: str,
    lesson_data: LessonCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add a lesson to a course"""
    course = await db.courses.find_one({"id": course_id}, {"_id": 0})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course["teacher_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    lesson = Lesson(course_id=course_id, **lesson_data.model_dump())
    await db.lessons.insert_one(serialize_doc(lesson.model_dump()))
    
    # Update course lesson count
    await db.courses.update_one(
        {"id": course_id},
        {"$inc": {"total_lessons": 1}}
    )
    
    return {"message": "Lesson created successfully", "lesson_id": lesson.id}

@api_router.get("/courses/{course_id}/lessons")
async def get_lessons(course_id: str):
    """Get all lessons for a course"""
    lessons = await db.lessons.find(
        {"course_id": course_id},
        {"_id": 0}
    ).sort("order", 1).to_list(100)
    return lessons

@api_router.put("/lessons/{lesson_id}")
async def update_lesson(
    lesson_id: str,
    lesson_data: LessonCreate,
    current_user: dict = Depends(get_current_user)
):
    """Update a lesson"""
    lesson = await db.lessons.find_one({"id": lesson_id}, {"_id": 0})
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    course = await db.courses.find_one({"id": lesson["course_id"]}, {"_id": 0})
    if course["teacher_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_data = lesson_data.model_dump()
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.lessons.update_one({"id": lesson_id}, {"$set": update_data})
    return {"message": "Lesson updated successfully"}

@api_router.delete("/lessons/{lesson_id}")
async def delete_lesson(
    lesson_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a lesson"""
    lesson = await db.lessons.find_one({"id": lesson_id}, {"_id": 0})
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    course = await db.courses.find_one({"id": lesson["course_id"]}, {"_id": 0})
    if course["teacher_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.lessons.delete_one({"id": lesson_id})
    await db.courses.update_one({"id": lesson["course_id"]}, {"$inc": {"total_lessons": -1}})
    return {"message": "Lesson deleted successfully"}

# ==================== ENROLLMENT ROUTES ====================

@api_router.post("/enrollments")
async def enroll_in_course(
    enrollment_data: EnrollmentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Enroll in a course"""
    if current_user["user_type"] != "student":
        raise HTTPException(status_code=403, detail="Only students can enroll")
    
    course = await db.courses.find_one({"id": enrollment_data.course_id}, {"_id": 0})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Check if already enrolled
    existing = await db.enrollments.find_one({
        "user_id": current_user["id"],
        "course_id": enrollment_data.course_id
    })
    if existing:
        raise HTTPException(status_code=400, detail="Already enrolled in this course")
    
    enrollment = Enrollment(
        user_id=current_user["id"],
        course_id=enrollment_data.course_id
    )
    await db.enrollments.insert_one(serialize_doc(enrollment.model_dump()))
    
    # Update course enrollment count
    await db.courses.update_one(
        {"id": enrollment_data.course_id},
        {"$inc": {"total_enrolled": 1}}
    )
    
    return {"message": "Enrolled successfully", "enrollment_id": enrollment.id}

@api_router.get("/enrollments/my")
async def get_my_enrollments(current_user: dict = Depends(get_current_user)):
    """Get student's enrollments"""
    enrollments = await db.enrollments.find(
        {"user_id": current_user["id"]},
        {"_id": 0}
    ).to_list(100)
    
    # Enrich with course data
    result = []
    for enrollment in enrollments:
        course = await db.courses.find_one({"id": enrollment["course_id"]}, {"_id": 0})
        if course:
            enrollment["course"] = course
        result.append(enrollment)
    
    return result

@api_router.put("/enrollments/{enrollment_id}/progress")
async def update_progress(
    enrollment_id: str,
    lesson_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Mark a lesson as completed"""
    enrollment = await db.enrollments.find_one({"id": enrollment_id}, {"_id": 0})
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    
    if enrollment["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    completed_lessons = enrollment.get("completed_lessons", [])
    if lesson_id not in completed_lessons:
        completed_lessons.append(lesson_id)
    
    # Calculate progress
    total_lessons = await db.lessons.count_documents({"course_id": enrollment["course_id"]})
    progress = (len(completed_lessons) / total_lessons * 100) if total_lessons > 0 else 0
    
    await db.enrollments.update_one(
        {"id": enrollment_id},
        {"$set": {
            "completed_lessons": completed_lessons,
            "progress": progress,
            "is_completed": progress >= 100,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    return {"message": "Progress updated", "progress": progress}

# ==================== BOOKING ROUTES ====================

@api_router.post("/bookings", response_model=BookingResponse)
async def create_booking(
    booking_data: BookingCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a booking with a teacher"""
    if current_user["user_type"] != "student":
        raise HTTPException(status_code=403, detail="Only students can create bookings")
    
    teacher_profile = await db.teacher_profiles.find_one({"user_id": booking_data.teacher_id}, {"_id": 0})
    if not teacher_profile:
        raise HTTPException(status_code=404, detail="Teacher not found")
    
    teacher = await db.users.find_one({"id": booking_data.teacher_id}, {"_id": 0})
    
    # Calculate price
    hourly_rate = teacher_profile.get("hourly_rate", 0)
    price = (booking_data.duration_minutes / 60) * hourly_rate
    
    booking = Booking(
        student_id=current_user["id"],
        teacher_id=booking_data.teacher_id,
        subject=booking_data.subject,
        booking_type=booking_data.booking_type,
        scheduled_date=booking_data.scheduled_date,
        duration_minutes=booking_data.duration_minutes,
        mode=booking_data.mode,
        notes=booking_data.notes,
        price=price,
        currency=teacher_profile.get("currency", "GBP")
    )
    
    await db.bookings.insert_one(serialize_doc(booking.model_dump()))
    
    return BookingResponse(
        id=booking.id,
        student_id=booking.student_id,
        student_name=current_user.get("full_name"),
        teacher_id=booking.teacher_id,
        teacher_name=teacher.get("full_name") if teacher else None,
        subject=booking.subject,
        booking_type=booking.booking_type,
        scheduled_date=booking.scheduled_date,
        duration_minutes=booking.duration_minutes,
        mode=booking.mode,
        notes=booking.notes,
        status=booking.status,
        price=booking.price,
        currency=booking.currency,
        created_at=booking.created_at
    )

@api_router.get("/bookings/my", response_model=List[BookingResponse])
async def get_my_bookings(current_user: dict = Depends(get_current_user)):
    """Get user's bookings (as student or teacher)"""
    if current_user["user_type"] == "student":
        query = {"student_id": current_user["id"]}
    elif current_user["user_type"] == "teacher":
        query = {"teacher_id": current_user["id"]}
    else:
        raise HTTPException(status_code=403, detail="Organizations cannot have bookings")
    
    bookings = await db.bookings.find(query, {"_id": 0}).sort("scheduled_date", -1).to_list(100)
    
    result = []
    for booking in bookings:
        student = await db.users.find_one({"id": booking["student_id"]}, {"_id": 0})
        teacher = await db.users.find_one({"id": booking["teacher_id"]}, {"_id": 0})
        
        scheduled_date = booking.get("scheduled_date")
        created_at = booking.get("created_at")
        if isinstance(scheduled_date, str):
            scheduled_date = datetime.fromisoformat(scheduled_date)
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        result.append(BookingResponse(
            id=booking["id"],
            student_id=booking["student_id"],
            student_name=student.get("full_name") if student else None,
            teacher_id=booking["teacher_id"],
            teacher_name=teacher.get("full_name") if teacher else None,
            subject=booking["subject"],
            booking_type=booking["booking_type"],
            scheduled_date=scheduled_date,
            duration_minutes=booking["duration_minutes"],
            mode=booking["mode"],
            notes=booking.get("notes"),
            status=booking["status"],
            price=booking.get("price", 0),
            currency=booking.get("currency", "GBP"),
            created_at=created_at
        ))
    
    return result

@api_router.put("/bookings/{booking_id}/status")
async def update_booking_status(
    booking_id: str,
    status: str,
    current_user: dict = Depends(get_current_user)
):
    """Update booking status (teacher only for confirm/complete)"""
    if status not in ["pending", "confirmed", "completed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # Check authorization
    if status == "cancelled":
        if booking["student_id"] != current_user["id"] and booking["teacher_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized")
    else:
        if booking["teacher_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Only teacher can update status")
    
    await db.bookings.update_one(
        {"id": booking_id},
        {"$set": {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    return {"message": f"Booking {status}"}

# ==================== JOB ROUTES ====================

@api_router.post("/jobs")
async def create_job(
    job_data: JobCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a job posting (organizations only)"""
    if current_user["user_type"] != "organization":
        raise HTTPException(status_code=403, detail="Only organizations can post jobs")
    
    job = Job(
        posted_by=current_user["id"],
        **job_data.model_dump()
    )
    
    await db.jobs.insert_one(serialize_doc(job.model_dump()))
    return {"message": "Job posted successfully", "job_id": job.id}

@api_router.get("/jobs")
async def get_jobs(
    job_type: Optional[str] = None,
    subject: Optional[str] = None,
    board: Optional[str] = None,
    location: Optional[str] = None,
    skip: int = 0,
    limit: int = 20
):
    """Get active job listings"""
    query = {"is_active": True}
    
    if job_type:
        query["job_type"] = job_type
    if subject:
        query["subjects"] = {"$regex": subject, "$options": "i"}
    if board:
        query["boards"] = {"$regex": board, "$options": "i"}
    if location:
        query["location"] = {"$regex": location, "$options": "i"}
    
    jobs = await db.jobs.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
    return jobs

@api_router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get single job details"""
    job = await db.jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get organization info
    org = await db.users.find_one({"id": job["posted_by"]}, {"_id": 0})
    org_profile = await db.organization_profiles.find_one({"user_id": job["posted_by"]}, {"_id": 0})
    
    job["organization"] = {
        "name": org_profile.get("organization_name") if org_profile else org.get("full_name") if org else None,
        "type": org_profile.get("organization_type") if org_profile else None
    }
    
    return job

@api_router.put("/jobs/{job_id}")
async def update_job(
    job_id: str,
    job_data: JobCreate,
    current_user: dict = Depends(get_current_user)
):
    """Update a job posting"""
    job = await db.jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["posted_by"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_data = job_data.model_dump()
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.jobs.update_one({"id": job_id}, {"$set": update_data})
    return {"message": "Job updated successfully"}

@api_router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a job posting"""
    job = await db.jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["posted_by"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.jobs.delete_one({"id": job_id})
    return {"message": "Job deleted successfully"}

# ==================== REVIEW ROUTES ====================

@api_router.post("/reviews")
async def create_review(
    review_data: ReviewCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a review for a teacher"""
    if current_user["user_type"] != "student":
        raise HTTPException(status_code=403, detail="Only students can write reviews")
    
    # Check if student has completed a booking with this teacher
    booking = await db.bookings.find_one({
        "student_id": current_user["id"],
        "teacher_id": review_data.teacher_id,
        "status": "completed"
    })
    if not booking:
        raise HTTPException(status_code=400, detail="You can only review teachers you've had lessons with")
    
    # Check if already reviewed
    existing = await db.reviews.find_one({
        "student_id": current_user["id"],
        "teacher_id": review_data.teacher_id
    })
    if existing:
        raise HTTPException(status_code=400, detail="You've already reviewed this teacher")
    
    review = Review(
        student_id=current_user["id"],
        teacher_id=review_data.teacher_id,
        rating=review_data.rating,
        comment=review_data.comment
    )
    
    await db.reviews.insert_one(serialize_doc(review.model_dump()))
    
    # Update teacher rating
    reviews = await db.reviews.find({"teacher_id": review_data.teacher_id}, {"_id": 0}).to_list(1000)
    avg_rating = sum(r["rating"] for r in reviews) / len(reviews)
    
    await db.teacher_profiles.update_one(
        {"user_id": review_data.teacher_id},
        {"$set": {"rating": avg_rating, "total_reviews": len(reviews)}}
    )
    
    return {"message": "Review submitted successfully"}

@api_router.get("/teachers/{teacher_id}/reviews")
async def get_teacher_reviews(teacher_id: str, skip: int = 0, limit: int = 20):
    """Get reviews for a teacher"""
    reviews = await db.reviews.find(
        {"teacher_id": teacher_id},
        {"_id": 0}
    ).skip(skip).limit(limit).to_list(limit)
    
    # Enrich with student names
    result = []
    for review in reviews:
        student = await db.users.find_one({"id": review["student_id"]}, {"_id": 0})
        review["student_name"] = student.get("full_name") if student else "Anonymous"
        result.append(review)
    
    return result

# ==================== INQUIRY/CONTACT ROUTES ====================

@api_router.post("/inquiries")
async def create_inquiry(inquiry_data: InquiryCreate):
    """Submit a contact inquiry"""
    inquiry = Inquiry(**inquiry_data.model_dump())
    await db.inquiries.insert_one(serialize_doc(inquiry.model_dump()))
    return {"message": "Inquiry submitted successfully", "inquiry_id": inquiry.id}

@api_router.get("/inquiries")
async def get_inquiries(
    status: Optional[str] = None,
    inquiry_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get inquiries (admin only - for now any authenticated user)"""
    query = {}
    if status:
        query["status"] = status
    if inquiry_type:
        query["inquiry_type"] = inquiry_type
    
    inquiries = await db.inquiries.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
    return inquiries

# ==================== BOARD/CATEGORY ROUTES ====================

@api_router.get("/boards")
async def get_boards():
    """Get all examination boards"""
    return {
        "uk": [
            {"id": "aqa", "name": "AQA", "full_name": "Assessment and Qualifications Alliance"},
            {"id": "edexcel", "name": "Edexcel", "full_name": "Edexcel (Pearson)"},
            {"id": "ocr_a", "name": "OCR A", "full_name": "Oxford Cambridge and RSA - A"},
            {"id": "ocr_b", "name": "OCR B", "full_name": "Oxford Cambridge and RSA - B"},
            {"id": "wjec", "name": "WJEC", "full_name": "Welsh Joint Education Committee"},
            {"id": "ocr_mei", "name": "OCR MEI", "full_name": "OCR Mathematics in Education and Industry"}
        ],
        "international": [
            {"id": "cie", "name": "CIE", "full_name": "Cambridge International Examinations"},
            {"id": "ib", "name": "IB", "full_name": "International Baccalaureate"},
            {"id": "ap", "name": "AP", "full_name": "Advanced Placement"},
            {"id": "edexcel_ial", "name": "Edexcel IAL", "full_name": "Edexcel International A-Level"}
        ]
    }

@api_router.get("/subjects")
async def get_subjects():
    """Get all available subjects"""
    return [
        "Mathematics", "Further Mathematics", "Physics", "Chemistry", "Biology",
        "English Language", "English Literature", "History", "Geography",
        "Economics", "Business Studies", "Accounting", "Computer Science",
        "Psychology", "Sociology", "Religious Studies", "Philosophy",
        "French", "Spanish", "German", "Arabic", "Urdu", "Mandarin",
        "Art & Design", "Music", "Drama", "Physical Education",
        "Law", "Politics", "Media Studies"
    ]

@api_router.get("/levels")
async def get_levels():
    """Get all available qualification levels"""
    return [
        {"id": "gcse", "name": "GCSE", "description": "General Certificate of Secondary Education"},
        {"id": "igcse", "name": "IGCSE", "description": "International GCSE"},
        {"id": "a_level", "name": "A-Level", "description": "Advanced Level"},
        {"id": "as_level", "name": "AS-Level", "description": "Advanced Subsidiary Level"},
        {"id": "ib_diploma", "name": "IB Diploma", "description": "International Baccalaureate Diploma"},
        {"id": "ap", "name": "AP", "description": "Advanced Placement"},
        {"id": "btec", "name": "BTEC", "description": "Business and Technology Education Council"},
        {"id": "primary", "name": "Primary", "description": "Primary School Level (KS1/KS2)"},
        {"id": "secondary", "name": "Secondary", "description": "Secondary School Level (KS3)"},
        {"id": "university", "name": "University", "description": "Higher Education"}
    ]

# ==================== SEARCH ROUTES ====================

@api_router.get("/search/teachers")
async def search_teachers(
    q: str = Query(..., min_length=1),
    skip: int = 0,
    limit: int = 20
):
    """Search teachers by name, subject, or location"""
    # Search in users and profiles
    teachers = await db.teacher_profiles.find(
        {
            "is_approved": True,
            "$or": [
                {"subjects": {"$regex": q, "$options": "i"}},
                {"location": {"$regex": q, "$options": "i"}},
                {"boards": {"$regex": q, "$options": "i"}}
            ]
        },
        {"_id": 0}
    ).skip(skip).limit(limit).to_list(limit)
    
    # Also search by user name
    users = await db.users.find(
        {
            "user_type": "teacher",
            "full_name": {"$regex": q, "$options": "i"}
        },
        {"_id": 0}
    ).to_list(100)
    
    user_ids = [u["id"] for u in users]
    name_profiles = await db.teacher_profiles.find(
        {"user_id": {"$in": user_ids}, "is_approved": True},
        {"_id": 0}
    ).to_list(100)
    
    # Combine and dedupe
    all_profiles = {p["id"]: p for p in teachers}
    for p in name_profiles:
        all_profiles[p["id"]] = p
    
    # Enrich with user data
    result = []
    for profile in list(all_profiles.values())[:limit]:
        user = await db.users.find_one({"id": profile["user_id"]}, {"_id": 0})
        result.append(TeacherProfileResponse(
            id=profile["id"],
            user_id=profile["user_id"],
            full_name=user.get("full_name") if user else None,
            email=user.get("email") if user else None,
            profile_image=user.get("profile_image") if user else None,
            bio=profile.get("bio"),
            subjects=profile.get("subjects", []),
            boards=profile.get("boards", []),
            qualifications=profile.get("qualifications", []),
            experience_years=profile.get("experience_years", 0),
            hourly_rate=profile.get("hourly_rate", 0),
            currency=profile.get("currency", "GBP"),
            teaching_modes=profile.get("teaching_modes", []),
            location=profile.get("location"),
            rating=profile.get("rating", 0),
            total_reviews=profile.get("total_reviews", 0),
            is_approved=profile.get("is_approved", False)
        ))
    
    return result

@api_router.get("/search/courses")
async def search_courses(
    q: str = Query(..., min_length=1),
    skip: int = 0,
    limit: int = 20
):
    """Search courses by title, subject, or board"""
    courses = await db.courses.find(
        {
            "is_published": True,
            "$or": [
                {"title": {"$regex": q, "$options": "i"}},
                {"subject": {"$regex": q, "$options": "i"}},
                {"board": {"$regex": q, "$options": "i"}},
                {"description": {"$regex": q, "$options": "i"}}
            ]
        },
        {"_id": 0}
    ).skip(skip).limit(limit).to_list(limit)
    
    result = []
    for course in courses:
        teacher = await db.users.find_one({"id": course["teacher_id"]}, {"_id": 0})
        created_at = course.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        result.append(CourseResponse(
            id=course["id"],
            teacher_id=course["teacher_id"],
            teacher_name=teacher.get("full_name") if teacher else None,
            title=course["title"],
            description=course["description"],
            board=course["board"],
            board_category=course["board_category"],
            subject=course["subject"],
            level=course["level"],
            topics=course.get("topics", []),
            is_free=course.get("is_free", True),
            price=course.get("price", 0),
            currency=course.get("currency", "GBP"),
            thumbnail=course.get("thumbnail"),
            total_lessons=course.get("total_lessons", 0),
            total_enrolled=course.get("total_enrolled", 0),
            rating=course.get("rating", 0),
            is_published=course.get("is_published", False),
            created_at=created_at
        ))
    
    return result

# ==================== STATISTICS ROUTES ====================

@api_router.get("/stats")
async def get_platform_stats():
    """Get platform statistics"""
    teachers_count = await db.teacher_profiles.count_documents({"is_approved": True})
    students_count = await db.users.count_documents({"user_type": "student"})
    courses_count = await db.courses.count_documents({"is_published": True})
    subjects = await get_subjects()
    
    return {
        "total_teachers": teachers_count or 9500,  # Default from website
        "total_students": students_count or 1000,
        "total_courses": courses_count,
        "subjects_covered": len(subjects) or 1500,
        "approval_rate": 55.1  # From website
    }

# ==================== ADMIN ROUTES ====================

@api_router.put("/admin/teachers/{teacher_id}/approve")
async def approve_teacher(
    teacher_id: str,
    approved: bool,
    current_user: dict = Depends(get_current_user)
):
    """Approve or reject a teacher (admin function)"""
    # In production, add proper admin check
    await db.teacher_profiles.update_one(
        {"user_id": teacher_id},
        {"$set": {"is_approved": approved, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": f"Teacher {'approved' if approved else 'rejected'}"}

@api_router.put("/admin/organizations/{org_id}/verify")
async def verify_organization(
    org_id: str,
    verified: bool,
    current_user: dict = Depends(get_current_user)
):
    """Verify or unverify an organization (admin function)"""
    await db.organization_profiles.update_one(
        {"user_id": org_id},
        {"$set": {"is_verified": verified, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": f"Organization {'verified' if verified else 'unverified'}"}

# ==================== HEALTH CHECK ====================

@api_router.get("/")
async def root():
    return {"message": "KnowledgeDottCom API is running", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}

# Include router
app.include_router(api_router)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
